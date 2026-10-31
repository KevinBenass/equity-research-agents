"""Run the research team on one question and persist everything needed to audit the run."""

from __future__ import annotations

import json
import sqlite3
import time
from contextlib import nullcontext
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.types import Command

from equity_agents.config import Settings
from equity_agents.graph import Deps, build_graph, new_run_id
from equity_agents.llm import LLM, Backend
from equity_agents.memory.store import MemoryStore
from equity_agents.tools.base import Tool
from equity_agents.tools.filings import FilingsSearch
from equity_agents.tools.xbrl import FinancialsDB
from equity_agents.tracing import Tracer


@dataclass
class RunOutput:
    run_id: str
    memo: str
    state: dict[str, Any]
    totals: dict[str, Any]
    latency_s: float
    run_dir: Path
    interrupted: bool = False


def make_deps(
    settings: Settings,
    run_id: str,
    backend: Backend | None = None,
    filings_tool: Tool | None = None,
    memory: MemoryStore | None = None,
    db: FinancialsDB | None = None,
    trace: bool = True,
) -> Deps:
    run_dir = settings.resolve(settings.paths.runs_dir) / run_id
    tracer = Tracer(run_id, run_dir / "trace.jsonl" if trace else None)
    return Deps(
        settings=settings,
        llm=LLM(settings.models.prices, tracer, backend),
        memory=memory or MemoryStore(settings.resolve(settings.paths.memory_db)),
        db=db
        or FinancialsDB(
            settings.resolve(settings.tools.xbrl_cache_dir),
            settings.universe.tickers,
            settings.tools.sql_row_limit,
        ),
        filings_tool=filings_tool or FilingsSearch(settings.filings_api_url).tool(),
    )


def _checkpointer(settings: Settings):
    if not settings.orchestration.human_review:
        return nullcontext(None)
    path = settings.resolve(settings.paths.checkpoints_db)
    path.parent.mkdir(parents=True, exist_ok=True)
    return nullcontext(SqliteSaver(sqlite3.connect(path, check_same_thread=False)))


def _persist(
    deps: Deps, run_id: str, question: str, state: dict[str, Any], latency: float
) -> RunOutput:
    run_dir = deps.settings.resolve(deps.settings.paths.runs_dir) / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    totals = deps.llm.tracer.totals()
    memo = state.get("memo", "")
    (run_dir / "memo.md").write_text(memo)
    findings = [f.model_dump() for f in state.get("findings", [])]
    (run_dir / "findings.json").write_text(json.dumps(findings, indent=2))
    review = state.get("review")
    summary = {
        "run_id": run_id,
        "question": question,
        "latency_s": round(latency, 1),
        **totals,
        "plan": [t.model_dump() for t in state.get("tasks", [])],
        "plan_rationale": state.get("plan_rationale", ""),
        "revisions": state.get("revisions", 0),
        "research_rounds": state.get("research_rounds", 0),
        "verdict": review.verdict if review else None,
        "final_issues": [i.model_dump() for i in review.issues] if review else [],
        "worker_steps": {r.task_id: r.steps for r in state.get("reports", [])},
        "n_findings": len(findings),
        "low_confidence_findings": sum(f["confidence"] == "low" for f in findings),
    }
    (run_dir / "summary.json").write_text(json.dumps(summary, indent=2, default=str))
    return RunOutput(
        run_id, memo, state, totals, latency, run_dir, interrupted="__interrupt__" in state
    )


def research(
    question: str, settings: Settings, run_id: str | None = None, deps: Deps | None = None
) -> RunOutput:
    run_id = run_id or new_run_id()
    deps = deps or make_deps(settings, run_id)
    settings = deps.settings
    t0 = time.perf_counter()
    with _checkpointer(settings) as cp:
        graph = build_graph(deps, checkpointer=cp)
        config = {"configurable": {"thread_id": run_id}, "recursion_limit": 60}
        state = graph.invoke({"run_id": run_id, "question": question}, config)
    return _persist(deps, run_id, question, state, time.perf_counter() - t0)


def resume(
    run_id: str, settings: Settings, approve: bool, feedback: str = "", deps: Deps | None = None
) -> RunOutput:
    """Continue a run paused at human review (requires orchestration.human_review)."""
    deps = deps or make_deps(settings, run_id)
    settings = deps.settings
    t0 = time.perf_counter()
    with _checkpointer(settings) as cp:
        graph = build_graph(deps, checkpointer=cp)
        config = {"configurable": {"thread_id": run_id}, "recursion_limit": 60}
        question = graph.get_state(config).values.get("question", "")
        state = graph.invoke(Command(resume={"approve": approve, "feedback": feedback}), config)
    return _persist(deps, run_id, question, state, time.perf_counter() - t0)
