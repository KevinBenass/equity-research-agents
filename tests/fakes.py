"""Test doubles: an XBRL fixture in SEC's companyfacts format and a scripted Claude backend.

The scripted backend reads the same requests the real API would receive (system prompt,
messages, tool results) and answers deterministically, so tests exercise the real graph,
tool loop, grounding checks and critic routing without network access.
"""

from __future__ import annotations

import csv
import io
import json
import re
from pathlib import Path
from typing import Any


def _obs(
    val: float,
    start: str | None,
    end: str,
    form: str = "10-K",
    filed: str = "2025-02-26",
    fp: str = "FY",
) -> dict:
    o = {
        "end": end,
        "val": val,
        "accn": "0001045810-25-000023",
        "fy": int(end[:4]),
        "fp": fp,
        "form": form,
        "filed": filed,
    }
    if start:
        o["start"] = start
    return o


def write_companyfacts(cache_dir: Path) -> None:
    cache_dir.mkdir(parents=True, exist_ok=True)
    nvda = {
        "entityName": "NVIDIA CORP",
        "facts": {
            "us-gaap": {
                "Revenues": {
                    "units": {
                        "USD": [
                            _obs(26_974_000_000, "2022-01-31", "2023-01-29", filed="2023-02-24"),
                            _obs(60_922_000_000, "2023-01-30", "2024-01-28", filed="2024-02-21"),
                            _obs(130_497_000_000, "2024-01-29", "2025-01-26"),
                            _obs(
                                35_082_000_000, "2024-10-28", "2025-01-26", fp="Q4"
                            ),  # quarterly: dropped
                            _obs(
                                26_044_000_000, "2024-01-29", "2024-04-28", form="10-Q", fp="Q1"
                            ),  # 10-Q: dropped
                            _obs(
                                60_900_000_000, "2023-01-30", "2024-01-28", filed="2024-01-01"
                            ),  # older filing: superseded
                        ]
                    }
                },
                "OperatingIncomeLoss": {
                    "units": {
                        "USD": [
                            _obs(32_972_000_000, "2023-01-30", "2024-01-28", filed="2024-02-21"),
                            _obs(81_453_000_000, "2024-01-29", "2025-01-26"),
                        ]
                    }
                },
                "Assets": {"units": {"USD": [_obs(111_601_000_000, None, "2025-01-26")]}},
            }
        },
    }
    msft = {
        "entityName": "MICROSOFT CORPORATION",
        "facts": {
            "us-gaap": {
                "RevenueFromContractWithCustomerExcludingAssessedTax": {
                    "units": {
                        "USD": [
                            _obs(211_915_000_000, "2022-07-01", "2023-06-30", filed="2023-07-27"),
                            _obs(245_122_000_000, "2023-07-01", "2024-06-30", filed="2024-07-30"),
                            _obs(281_724_000_000, "2024-07-01", "2025-06-30", filed="2025-07-30"),
                        ]
                    }
                },
                "Revenues": {
                    "units": {"USD": [_obs(1.0, "2024-07-01", "2025-06-30")]}
                },  # lower priority concept
                "OperatingIncomeLoss": {
                    "units": {
                        "USD": [
                            _obs(88_523_000_000, "2022-07-01", "2023-06-30", filed="2023-07-27"),
                            _obs(109_433_000_000, "2023-07-01", "2024-06-30", filed="2024-07-30"),
                            _obs(128_528_000_000, "2024-07-01", "2025-06-30", filed="2025-07-30"),
                        ]
                    }
                },
            }
        },
    }
    (cache_dir / "NVDA.json").write_text(json.dumps(nvda))
    (cache_dir / "MSFT.json").write_text(json.dumps(msft))


PASSAGES = {
    "nvidia": '<passage ref="filing:NVDA-10-K-FY2025::structure::0042" company="NVIDIA Corp" fiscal_year="2025" '
    'section="Item 7. MD&A">Revenue growth was driven by Data Center, as demand for our Hopper '
    "computing platform used for large language models increased.</passage>",
}


def _csv_rows(text: str) -> list[dict[str, str]]:
    return list(csv.DictReader(io.StringIO(text)))


def _tool_results(messages: list[dict]) -> list[str]:
    out = []
    for m in messages:
        if m["role"] == "user" and isinstance(m["content"], list):
            out += [b["content"] for b in m["content"] if b.get("type") == "tool_result"]
    return out


def _use(name: str, args: dict, idx: int = 0) -> dict:
    return {
        "content": [{"type": "tool_use", "id": f"tu_{name}_{idx}", "name": name, "input": args}],
        "stop_reason": "tool_use",
        "usage": {"input_tokens": 800, "output_tokens": 120},
    }


def _text(t: str) -> dict:
    return {
        "content": [{"type": "text", "text": t}],
        "stop_reason": "end_turn",
        "usage": {"input_tokens": 1500, "output_tokens": 400},
    }


class ScriptedBackend:
    """Plays every role for the question 'How did NVIDIA's revenue change in fiscal 2025 and why?'."""

    def __init__(
        self,
        first_draft_wrong: bool = False,
        quant_lies: bool = False,
        critic_verdict: str = "approve",
    ) -> None:
        self.first_draft_wrong = first_draft_wrong
        self.quant_lies = quant_lies
        self.critic_verdict = critic_verdict
        self.requests: list[dict[str, Any]] = []
        self.writer_calls = 0

    def create(self, **req: Any) -> dict[str, Any]:
        self.requests.append(req)
        system = req["system"][0]["text"]
        msgs = req["messages"]
        if "lead analyst" in system:
            return _use(
                "submit_plan",
                {
                    "rationale": "Numbers from XBRL, explanation from the 10-K.",
                    "tasks": [
                        {
                            "kind": "quant",
                            "instruction": "NVIDIA (NVDA) revenue FY2024 and FY2025 and growth %",
                            "tickers": ["NVDA"],
                            "fiscal_years": [2024, 2025],
                        },
                        {
                            "kind": "filings",
                            "instruction": "What drove NVIDIA's FY2025 revenue growth per its 10-K?",
                            "tickers": ["NVDA"],
                            "fiscal_years": [2025],
                        },
                    ],
                },
            )
        if "quantitative analyst" in system:
            results = _tool_results(msgs)
            if not results:
                return _use(
                    "get_metric",
                    {"ticker": "NVDA", "metric": "revenue", "fiscal_years": [2024, 2025]},
                )
            if len(results) == 1:
                rows = {int(r["fiscal_year"]): float(r["value"]) for r in _csv_rows(results[0])}
                return _use(
                    "calculate",
                    {"expression": f"({rows[2025]} - {rows[2024]}) / {rows[2024]} * 100"},
                    1,
                )
            rows = {
                int(r["fiscal_year"]): (float(r["value"]), r["ref"]) for r in _csv_rows(results[0])
            }
            growth = float(results[1])
            expr = f"({rows[2025][0]} - {rows[2024][0]}) / {rows[2024][0]} * 100"
            v25 = rows[2025][0] * (1.1 if self.quant_lies else 1)
            return _use(
                "submit_findings",
                {
                    "findings": [
                        {
                            "claim": "NVIDIA revenue FY2025",
                            "ticker": "NVDA",
                            "metric": "revenue",
                            "value": v25,
                            "unit": "USD",
                            "period": "FY2025",
                            "source": {"kind": "xbrl", "ref": rows[2025][1]},
                        },
                        {
                            "claim": "NVIDIA revenue FY2024",
                            "ticker": "NVDA",
                            "metric": "revenue",
                            "value": rows[2024][0],
                            "unit": "USD",
                            "period": "FY2024",
                            "source": {"kind": "xbrl", "ref": rows[2024][1]},
                        },
                        {
                            "claim": f"Revenue growth FY2025 vs FY2024 from {rows[2024][1]} and {rows[2025][1]}",
                            "ticker": "NVDA",
                            "metric": "revenue_growth",
                            "value": round(growth, 1),
                            "unit": "%",
                            "period": "FY2025",
                            "source": {"kind": "calc", "ref": f"calc:{expr}"},
                        },
                    ]
                },
                2,
            )
        if "filings researcher" in system:
            results = _tool_results(msgs)
            if not results:
                return _use(
                    "search_filings", {"query": "NVIDIA fiscal 2025 revenue growth drivers"}
                )
            ref = re.search(r'ref="([^"]+)"', results[0]).group(1)
            return _use(
                "submit_findings",
                {
                    "findings": [
                        {
                            "claim": "Management attributes FY2025 revenue growth to Data Center demand for Hopper platforms used for LLMs.",
                            "ticker": "NVDA",
                            "source": {
                                "kind": "filing",
                                "ref": ref,
                                "quote": "Revenue growth was driven by Data Center",
                            },
                        },
                    ]
                },
                1,
            )
        if "writer on an equity" in system:
            self.writer_calls += 1
            user = msgs[0]["content"]
            fs = [
                json.loads(line)
                for line in user.split("Findings (JSON lines):\n")[1].split("\n\n")[0].splitlines()
                if line.startswith("{")
            ]
            by = {f.get("source"): f for f in fs}
            rev = [f for f in fs if f.get("unit") == "USD"]
            cur = max(rev, key=lambda f: f["period"])
            prev = min(rev, key=lambda f: f["period"])
            growth = next(f for f in fs if f.get("unit") == "%")
            driver = next(f for f in by.values() if "value" not in f)
            cur_b = cur["value"] / 1e9 + (
                1.4 if self.first_draft_wrong and self.writer_calls == 1 else 0
            )
            return _text(
                "# NVIDIA's revenue more than doubled in fiscal 2025\n## Summary\n"
                f"NVIDIA's revenue rose to ${cur_b:.1f} billion in fiscal 2025 [{cur['id']}] from "
                f"${prev['value'] / 1e9:.1f} billion in fiscal 2024 [{prev['id']}], an increase of {growth['value']:.1f}% [{growth['id']}].\n"
                f"## Analysis\nManagement attributes the growth to Data Center demand for Hopper systems used to train large language models [{driver['id']}].\n"
                "## Risks and caveats\nThe explanation rests on management's own account in the 10-K."
            )
        if "reviewer on an equity" in system:
            return _use("submit_review", {"verdict": self.critic_verdict, "issues": []})
        raise AssertionError(f"Unexpected system prompt: {system[:80]}")
