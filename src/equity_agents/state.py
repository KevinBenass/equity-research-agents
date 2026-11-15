"""Shared state and the structured handoff contracts between agents.

Agents never pass prose transcripts to each other. Researchers return ``Finding`` objects
(a claim, an optional number with unit and period, and a source); the writer may only use
findings; the critic checks the memo against them. This keeps each agent's context small
and makes every number in the final memo traceable to a tool call.
"""

from __future__ import annotations

import operator
from typing import Annotated, Literal, TypedDict

from pydantic import BaseModel, Field

TaskKind = Literal["filings", "quant"]


class Task(BaseModel):
    id: str
    kind: TaskKind
    instruction: str
    tickers: list[str] = Field(default_factory=list)
    fiscal_years: list[int] = Field(default_factory=list)


class Source(BaseModel):
    kind: Literal["xbrl", "filing", "calc", "memory"]
    ref: str  # e.g. "xbrl:NVDA:revenue:FY2025", "filing:NVDA-10-K-FY2025::structure::0042"
    quote: str = ""
    # Set by code, never by the model: the quote was found verbatim in a tool result.
    quote_verified: bool = False


class Finding(BaseModel):
    id: str = ""  # assigned by the graph when merged ("F1", "F2", ...)
    task_id: str = ""
    claim: str
    ticker: str | None = None
    metric: str | None = None
    value: float | None = None
    unit: str | None = None  # "USD", "USD/share", "%", "ratio", ...
    period: str | None = None  # "FY2025"
    source: Source
    confidence: Literal["high", "medium", "low"] = "high"


class Issue(BaseModel):
    kind: Literal[
        "unsupported_number",
        "missing_citation",
        "unknown_citation",
        "unsupported_claim",
        "missing_evidence",
        "other",
    ]
    detail: str
    excerpt: str = ""


class Review(BaseModel):
    verdict: Literal["approve", "revise", "research_more"]
    issues: list[Issue] = Field(default_factory=list)
    gap_tasks: list[Task] = Field(default_factory=list)


class WorkerReport(BaseModel):
    task_id: str
    findings: list[Finding]
    steps: int
    tool_calls: list[str]
    notes: str = ""


class ResearchState(TypedDict, total=False):
    run_id: str
    question: str
    memory_context: str  # compact digest of semantic + episodic memory for this question
    plan_rationale: str
    tasks: list[Task]  # tasks of the current research round
    reports: Annotated[list[WorkerReport], operator.add]  # fan-in from parallel workers
    findings: list[Finding]  # merged, de-duplicated, numbered
    memo: str
    review: Review
    revisions: int
    research_rounds: int
    human_feedback: str
    approved: bool
