"""Evaluation questions with ground truth computed from XBRL, not written by hand or by an LLM.

Templates cover the shapes of real analyst requests. Each item carries the numbers a correct
memo must contain and the specialist types the plan must involve.
"""

from __future__ import annotations

import json
import random
from pathlib import Path

from pydantic import BaseModel, Field

from equity_agents.tools.xbrl import FinancialsDB

LABEL = {"revenue": "revenue", "net_income": "net income", "operating_income": "operating income"}


class TruthNumber(BaseModel):
    label: str
    value: float
    unit: str  # "USD" or "%"


class EvalItem(BaseModel):
    qid: str
    question: str
    template: str
    tickers: list[str]
    truth: list[TruthNumber]
    requires: list[str] = Field(default_factory=list)  # specialist kinds: quant, filings


def _short(name: str) -> str:
    for suffix in (", Inc.", " Inc.", " Corporation", " Corp", " & Co.", " Co.", ", Inc"):
        name = name.replace(suffix, "")
    return name.strip().title() if name.isupper() else name.strip()


def build_eval_set(
    db: FinancialsDB, years: list[int], seed: int = 0, n_per_template: int = 6
) -> list[EvalItem]:
    rng = random.Random(seed)
    tickers = sorted(db.df["ticker"].unique())
    latest = [y for y in years if y - 1 in years]
    items: list[EvalItem] = []

    def growth(t: str, metric: str, fy: int) -> tuple[float, float, float] | None:
        cur, prev = db.get(t, metric, fy), db.get(t, metric, fy - 1)
        if cur is None or prev is None or prev == 0:
            return None
        return cur, prev, (cur - prev) / abs(prev) * 100

    combos = [(t, m, y) for t in tickers for m in LABEL for y in latest]
    rng.shuffle(combos)
    for t, metric, fy in combos:
        if sum(i.template == "growth" for i in items) >= n_per_template:
            break
        g = growth(t, metric, fy)
        if g:
            name = _short(db.names.get(t, t))
            items.append(
                EvalItem(
                    qid=f"growth-{t}-{metric}-{fy}",
                    template="growth",
                    tickers=[t],
                    requires=["quant"],
                    question=f"How much did {name}'s {LABEL[metric]} change in fiscal {fy} compared with fiscal {fy - 1}?",
                    truth=[
                        TruthNumber(label=f"{metric} FY{fy}", value=g[0], unit="USD"),
                        TruthNumber(label=f"{metric} FY{fy - 1}", value=g[1], unit="USD"),
                        TruthNumber(label="growth", value=g[2], unit="%"),
                    ],
                )
            )

    pairs = [(t, y) for t in tickers for y in years]
    rng.shuffle(pairs)
    for t, fy in pairs:
        if sum(i.template == "margin" for i in items) >= n_per_template:
            break
        rev, op = db.get(t, "revenue", fy), db.get(t, "operating_income", fy)
        if rev and op:
            name = _short(db.names.get(t, t))
            items.append(
                EvalItem(
                    qid=f"margin-{t}-{fy}",
                    template="margin",
                    tickers=[t],
                    requires=["quant"],
                    question=f"What was {name}'s operating margin in fiscal {fy}?",
                    truth=[TruthNumber(label="operating margin", value=op / rev * 100, unit="%")],
                )
            )

    for _ in range(50):
        if sum(i.template == "comparison" for i in items) >= n_per_template or len(tickers) < 2:
            break
        a, b = rng.sample(tickers, 2)
        fy = rng.choice(latest) if latest else None
        ga, gb = (growth(a, "revenue", fy), growth(b, "revenue", fy)) if fy else (None, None)
        qid = f"compare-{a}-{b}-{fy}"
        if ga and gb and qid not in {i.qid for i in items}:
            na, nb = _short(db.names.get(a, a)), _short(db.names.get(b, b))
            items.append(
                EvalItem(
                    qid=qid,
                    template="comparison",
                    tickers=[a, b],
                    requires=["quant"],
                    question=f"Compare the revenue growth of {na} and {nb} in fiscal {fy}.",
                    truth=[
                        TruthNumber(label=f"{a} growth", value=ga[2], unit="%"),
                        TruthNumber(label=f"{b} growth", value=gb[2], unit="%"),
                    ],
                )
            )

    rng.shuffle(combos)
    for t, metric, fy in combos:
        if sum(i.template == "drivers" for i in items) >= n_per_template:
            break
        if metric != "revenue":
            continue
        g = growth(t, metric, fy)
        if g:
            name = _short(db.names.get(t, t))
            items.append(
                EvalItem(
                    qid=f"drivers-{t}-{fy}",
                    template="drivers",
                    tickers=[t],
                    requires=["quant", "filings"],
                    question=f"How did {name}'s revenue change in fiscal {fy}, and what does management say drove it?",
                    truth=[TruthNumber(label="growth", value=g[2], unit="%")],
                )
            )
    return items


def save_eval_set(items: list[EvalItem], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(i.model_dump()) + "\n" for i in items))


def load_eval_set(path: Path) -> list[EvalItem]:
    return [EvalItem.model_validate_json(x) for x in path.read_text().splitlines() if x.strip()]
