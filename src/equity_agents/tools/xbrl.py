"""Structured financials from SEC XBRL "company facts", queryable with SQL in DuckDB.

Numbers in a research memo should come from structured data, not from an LLM reading a
table in a PDF. Each company's facts are downloaded once from data.sec.gov, normalized to a
small set of canonical metrics, and loaded into one DuckDB table:

    financials(ticker, metric, fiscal_year, value, unit, period_end, concept, accession, filed)

Normalization rules (where most XBRL pitfalls are):
- Companies tag the same line item with different concepts (revenue alone has several);
  each canonical metric has a priority list and the first concept reported wins.
- Annual flow values (income, cash flow) must span ~1 year; quarterly values are dropped.
- The same period is re-reported in later filings (restatements); the latest filing wins.
- Fiscal year = calendar year of the period end, as in the filings-copilot corpus.
"""

from __future__ import annotations

import json
import re
import threading
from datetime import date
from pathlib import Path

import duckdb
import httpx
import pandas as pd

from equity_agents.tools.base import Tool, ToolError

METRICS: dict[str, dict] = {
    "revenue": {
        "concepts": [
            "RevenueFromContractWithCustomerExcludingAssessedTax",
            "Revenues",
            "SalesRevenueNet",
            "RevenuesNetOfInterestExpense",
        ],
        "unit": "USD",
        "flow": True,
    },
    "gross_profit": {"concepts": ["GrossProfit"], "unit": "USD", "flow": True},
    "operating_income": {"concepts": ["OperatingIncomeLoss"], "unit": "USD", "flow": True},
    "net_income": {"concepts": ["NetIncomeLoss"], "unit": "USD", "flow": True},
    "rnd_expense": {"concepts": ["ResearchAndDevelopmentExpense"], "unit": "USD", "flow": True},
    "eps_diluted": {"concepts": ["EarningsPerShareDiluted"], "unit": "USD/shares", "flow": True},
    "operating_cash_flow": {
        "concepts": ["NetCashProvidedByUsedInOperatingActivities"],
        "unit": "USD",
        "flow": True,
    },
    "capex": {
        "concepts": [
            "PaymentsToAcquirePropertyPlantAndEquipment",
            "PaymentsToAcquireProductiveAssets",
        ],
        "unit": "USD",
        "flow": True,
    },
    "total_assets": {"concepts": ["Assets"], "unit": "USD", "flow": False},
    "cash": {"concepts": ["CashAndCashEquivalentsAtCarryingValue"], "unit": "USD", "flow": False},
    "long_term_debt": {
        "concepts": ["LongTermDebtNoncurrent", "LongTermDebt"],
        "unit": "USD",
        "flow": False,
    },
    "shareholders_equity": {"concepts": ["StockholdersEquity"], "unit": "USD", "flow": False},
}

TICKERS_URL = "https://www.sec.gov/files/company_tickers.json"
FACTS_URL = "https://data.sec.gov/api/xbrl/companyfacts/CIK{cik}.json"


def download_facts(
    tickers: list[str], cache_dir: Path, user_agent: str, ciks: dict[str, str] | None = None
) -> None:
    cache_dir.mkdir(parents=True, exist_ok=True)
    missing = [t for t in tickers if not (cache_dir / f"{t}.json").exists()]
    if not missing:
        return
    if "@" not in user_agent:
        raise ValueError("Set SEC_USER_AGENT='Your Name you@example.com' (required by the SEC).")
    with httpx.Client(
        headers={"User-Agent": user_agent}, timeout=60, follow_redirects=True
    ) as client:
        cik_map = {t: str(c).zfill(10) for t, c in (ciks or {}).items()}
        if any(t not in cik_map for t in missing):
            for r in client.get(TICKERS_URL).json().values():
                cik_map.setdefault(r["ticker"].upper(), str(r["cik_str"]).zfill(10))
        for t in missing:
            if t not in cik_map:
                raise ValueError(f"No SEC CIK for {t}; add it to universe.ciks in the config.")
            resp = client.get(FACTS_URL.format(cik=cik_map[t]))
            resp.raise_for_status()
            (cache_dir / f"{t}.json").write_text(resp.text)


def normalize_facts(ticker: str, facts: dict) -> pd.DataFrame:
    gaap = facts.get("facts", {}).get("us-gaap", {})
    rows = []
    for metric, spec in METRICS.items():
        chosen: dict[int, dict] = {}
        for priority, concept in enumerate(spec["concepts"]):
            units = gaap.get(concept, {}).get("units", {})
            for obs in units.get(spec["unit"], []):
                if obs.get("form") not in ("10-K", "10-K/A"):
                    continue
                end = date.fromisoformat(obs["end"])
                if spec["flow"]:
                    if "start" not in obs:
                        continue
                    days = (end - date.fromisoformat(obs["start"])).days
                    if not 330 <= days <= 380:
                        continue
                fy = end.year
                current = chosen.get(fy)
                key = (-priority, obs.get("filed", ""))
                if current is None or key > current["_key"]:
                    chosen[fy] = {**obs, "_key": key, "_concept": concept}
        for fy, obs in chosen.items():
            rows.append(
                {
                    "ticker": ticker,
                    "metric": metric,
                    "fiscal_year": fy,
                    "value": float(obs["val"]),
                    "unit": spec["unit"],
                    "period_end": obs["end"],
                    "concept": obs["_concept"],
                    "accession": obs.get("accn", ""),
                    "filed": obs.get("filed", ""),
                }
            )
    return pd.DataFrame(rows)


class FinancialsDB:
    def __init__(self, cache_dir: Path, tickers: list[str], row_limit: int = 200) -> None:
        frames = []
        self.names: dict[str, str] = {}
        for t in tickers:
            path = cache_dir / f"{t}.json"
            if path.exists():
                facts = json.loads(path.read_text())
                self.names[t] = facts.get("entityName", t)
                frames.append(normalize_facts(t, facts))
        self.df = (
            pd.concat(frames, ignore_index=True)
            if frames
            else pd.DataFrame(
                columns=[
                    "ticker",
                    "metric",
                    "fiscal_year",
                    "value",
                    "unit",
                    "period_end",
                    "concept",
                    "accession",
                    "filed",
                ]
            )
        )
        self.row_limit = row_limit
        self.con = duckdb.connect()
        self.con.register("financials", self.df)
        # Quant workers run in parallel threads and share this connection; DuckDB
        # connections are not safe for concurrent use, so queries are serialized.
        self._lock = threading.Lock()

    def get(self, ticker: str, metric: str, fiscal_year: int) -> float | None:
        m = self.df[
            (self.df.ticker == ticker)
            & (self.df.metric == metric)
            & (self.df.fiscal_year == fiscal_year)
        ]
        return float(m["value"].iloc[0]) if len(m) else None

    # ------------------------------------------------------------------ tools
    def list_metrics(self, ticker: str | None = None) -> str:
        df = self.df if not ticker else self.df[self.df.ticker == ticker.upper()]
        if df.empty:
            raise ToolError(
                f"No data for ticker {ticker!r}. Available: {sorted(self.df.ticker.unique())}"
            )
        cov = (
            df.groupby(["ticker", "metric"])["fiscal_year"]
            .agg(lambda s: f"{s.min()}-{s.max()}")
            .reset_index()
        )
        return cov.to_csv(index=False)

    def get_metric(self, ticker: str, metric: str, fiscal_years: list[int] | None = None) -> str:
        if metric not in METRICS:
            raise ToolError(f"Unknown metric {metric!r}. Options: {sorted(METRICS)}")
        df = self.df[(self.df.ticker == ticker.upper()) & (self.df.metric == metric)]
        if fiscal_years:
            df = df[df.fiscal_year.isin(fiscal_years)]
        if df.empty:
            raise ToolError(f"No {metric} for {ticker} in {fiscal_years or 'any year'}")
        out = df.sort_values("fiscal_year")[
            ["ticker", "metric", "fiscal_year", "value", "unit", "period_end", "accession"]
        ]
        return out.assign(
            ref=[f"xbrl:{r.ticker}:{r.metric}:FY{r.fiscal_year}" for r in out.itertuples()]
        ).to_csv(index=False)

    def query(self, sql: str) -> str:
        """Read-only SQL over the financials table."""
        cleaned = sql.strip().rstrip(";")
        if not re.match(r"(?is)^\s*(select|with)\b", cleaned) or re.search(
            r"(?i)\b(insert|update|delete|drop|create|alter|attach|copy|pragma|install|load)\b",
            cleaned,
        ):
            raise ToolError("Only read-only SELECT queries on `financials` are allowed.")
        if re.search(r"(?i)read_\w+\s*\(|\bhttp", cleaned):
            raise ToolError("External reads are not allowed; query the `financials` table.")
        try:
            with self._lock:
                df = self.con.execute(f"SELECT * FROM ({cleaned}) LIMIT {self.row_limit}").df()
        except duckdb.Error as exc:
            raise ToolError(f"SQL error: {exc}") from exc
        return df.to_csv(index=False) if len(df) else "(no rows)"

    def tools(self) -> list[Tool]:
        return [
            Tool(
                "list_metrics",
                "List which canonical metrics and fiscal years are available, optionally for one ticker.",
                {"type": "object", "properties": {"ticker": {"type": "string"}}},
                self.list_metrics,
            ),
            Tool(
                "get_metric",
                "Annual values of one canonical metric for one company from SEC XBRL filings. Metrics: "
                + ", ".join(METRICS)
                + ". Values are in raw units (USD, not millions). Each row has a `ref` to cite.",
                {
                    "type": "object",
                    "properties": {
                        "ticker": {"type": "string"},
                        "metric": {"type": "string"},
                        "fiscal_years": {"type": "array", "items": {"type": "integer"}},
                    },
                    "required": ["ticker", "metric"],
                },
                self.get_metric,
            ),
            Tool(
                "query_financials",
                "Read-only DuckDB SQL over table financials(ticker, metric, fiscal_year, value, unit, period_end, "
                "concept, accession, filed). Use for cross-company or multi-metric questions.",
                {"type": "object", "properties": {"sql": {"type": "string"}}, "required": ["sql"]},
                self.query,
            ),
        ]
