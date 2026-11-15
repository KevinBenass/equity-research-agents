"""The research team as a LangGraph state machine.

    recall ─► planner ─┬─► worker (quant)    ─┐
                       ├─► worker (filings)  ─┼─► merge ─► writer ─► critic ─┬─► finalize
                       └─► worker (…)        ─┘      ▲        ▲              │
                                                     │        └── revise ────┤
                                                     └── research_more ──────┤
                                                              human_review ◄─┘ (optional)

- Fan-out uses LangGraph ``Send``: one worker per task, run in parallel, results fan in
  through an additive reducer.
- Loops are bounded (max revisions, max research rounds), so a run always terminates.
- With a checkpointer, a run can pause at human_review (``interrupt``) and resume later.
"""

from __future__ import annotations

import re
import uuid
from dataclasses import dataclass, field
from typing import Any

from langgraph.graph import END, START, StateGraph
from langgraph.types import Send, interrupt

from equity_agents.agents import prompts
from equity_agents.agents.worker import findings_digest, run_worker
from equity_agents.config import Settings
from equity_agents.llm import LLM, text_of
from equity_agents.memory.store import MemoryStore
from equity_agents.numbers import citation_problems, unsupported_numbers
from equity_agents.state import Finding, Issue, ResearchState, Review, Task
from equity_agents.tools.base import Tool
from equity_agents.tools.calc import CALC_TOOL
from equity_agents.tools.xbrl import METRICS, FinancialsDB

ALIASES = {
    "AAPL": ["apple"],
    "MSFT": ["microsoft"],
    "NVDA": ["nvidia"],
    "AMZN": ["amazon", "aws"],
    "GOOGL": ["alphabet", "google"],
    "JPM": ["jpmorgan", "jp morgan", "chase"],
    "JNJ": ["johnson & johnson", "j&j"],
    "XOM": ["exxon", "exxonmobil"],
    "META": ["meta platforms"],
    "TSLA": ["tesla"],
}


def mentioned_tickers(text: str, universe: list[str]) -> list[str]:
    low = text.lower()
    out = []
    for t in universe:
        names = [t.lower(), *ALIASES.get(t, [])]
        if any(re.search(rf"(?<![a-z0-9]){re.escape(n)}(?![a-z0-9])", low) for n in names):
            out.append(t)
    return out


@dataclass
class Deps:
    settings: Settings
    llm: LLM
    memory: MemoryStore
    db: FinancialsDB
    filings_tool: Tool
    extra: dict[str, Any] = field(default_factory=dict)


# Structured-output schemas (output_config.format). The API requires every object to set
# additionalProperties false and list all its properties as required; no length constraints.
TASK_SCHEMA = {
    "type": "object",
    "properties": {
        "kind": {"type": "string", "enum": ["quant", "filings"]},
        "instruction": {"type": "string"},
        "tickers": {"type": "array", "items": {"type": "string"}},
        "fiscal_years": {"type": "array", "items": {"type": "integer"}},
    },
    "required": ["kind", "instruction", "tickers", "fiscal_years"],
    "additionalProperties": False,
}
PLAN_SCHEMA = {
    "type": "object",
    "properties": {
        "rationale": {"type": "string"},
        "tasks": {"type": "array", "items": TASK_SCHEMA},
    },
    "required": ["rationale", "tasks"],
    "additionalProperties": False,
}
REVIEW_SCHEMA = {
    "type": "object",
    "properties": {
        "verdict": {"type": "string", "enum": ["approve", "revise", "research_more"]},
        "issues": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "kind": {
                        "type": "string",
                        "enum": ["unsupported_claim", "missing_evidence", "other"],
                    },
                    "detail": {"type": "string"},
                    "excerpt": {"type": "string"},
                },
                "required": ["kind", "detail", "excerpt"],
                "additionalProperties": False,
            },
        },
        "gap_tasks": {"type": "array", "items": TASK_SCHEMA},
    },
    "required": ["verdict", "issues", "gap_tasks"],
    "additionalProperties": False,
}


def build_graph(deps: Deps, checkpointer: Any = None):
    s, llm, mem = deps.settings, deps.llm, deps.memory
    o, ctx = s.orchestration, s.context

    def make_tasks(raw: list[dict], prefix: str) -> list[Task]:
        return _make_tasks(raw, prefix, s)

    # ------------------------------------------------------------------ nodes
    def recall(state: ResearchState) -> dict:
        tickers = mentioned_tickers(state["question"], s.universe.tickers)
        facts = mem.recall_facts(tickers or None, limit=ctx.memory_facts_in_prompt)
        episodes = mem.similar_episodes(state["question"], k=ctx.episodes_in_prompt)
        parts = []
        if facts:
            parts.append(
                "Facts in memory:\n"
                + "\n".join(
                    f"- {f['ticker']} {f['metric'] or 'note'} {f['period'] or ''}: "
                    f"{f['value'] if f['value'] is not None else f['claim'][:160]} ({f['ref']})"
                    for f in facts
                )
            )
        if episodes:
            parts.append(
                "Related past research:\n"
                + "\n".join(
                    f"- [{e['created'][:10]}] {e['question']} -> {e['summary'][:300]}"
                    for e in episodes
                )
            )
        return {
            "memory_context": "\n\n".join(parts),
            "revisions": 0,
            "research_rounds": 0,
            "findings": [],
        }

    def planner(state: ResearchState) -> dict:
        system = prompts.PLANNER.format(
            max_tasks=o.max_tasks,
            tickers=", ".join(s.universe.tickers),
            years=", ".join(map(str, s.universe.fiscal_years)),
        )
        user = f"Client question: {state['question']}"
        if state.get("memory_context"):
            user += f"\n\n{state['memory_context']}"
        plan = llm.structured(
            agent="planner",
            model=s.models.planner,
            system=system,
            messages=[{"role": "user", "content": user}],
            schema=PLAN_SCHEMA,
            max_tokens=1500,
        )
        return {
            "tasks": make_tasks(plan.get("tasks", []), "T"),
            "plan_rationale": plan.get("rationale", ""),
        }

    def fan_out(state: ResearchState) -> list[Send]:
        return [Send("worker", {"task": t}) for t in state["tasks"]] or [Send("merge", state)]

    def worker(payload: dict) -> dict:
        task: Task = payload["task"]
        facts = mem.recall_facts(task.tickers or None, limit=ctx.memory_facts_in_prompt)
        if task.kind == "quant":
            mem_tool = Tool(
                "recall_facts",
                "Facts saved from earlier research (with refs). Check before querying.",
                {
                    "type": "object",
                    "properties": {"ticker": {"type": "string"}, "metric": {"type": "string"}},
                },
                lambda ticker=None, metric=None: (
                    mem.recall_facts([ticker] if ticker else None, metric) or "nothing in memory"
                ),
            )
            tools, system, model = (
                [*deps.db.tools(), CALC_TOOL, mem_tool],
                prompts.QUANT,
                s.models.quant_analyst,
            )
            facts = [f for f in facts if f["metric"] in METRICS]
        else:
            tools, system, model = [deps.filings_tool], prompts.FILINGS, s.models.filings_researcher
            facts = [f for f in facts if f["metric"] in (None, "claim")]
        report = run_worker(
            task,
            llm=llm,
            model=model,
            system=system,
            tools=tools,
            facts=facts,
            max_steps=o.max_worker_steps,
            budget_tokens=ctx.worker_budget_tokens,
            keep_recent=ctx.keep_recent_tool_results,
            max_chars=ctx.tool_result_max_chars,
            db=deps.db if task.kind == "quant" else None,
        )
        return {"reports": [report]}

    def merge(state: ResearchState) -> dict:
        current = {t.id for t in state.get("tasks", [])}
        existing = list(state.get("findings", []))
        seen = {(f.ticker, f.metric, f.period, f.value, f.claim) for f in existing}
        n = len(existing)
        for report in state.get("reports", []):
            if report.task_id not in current:
                continue
            for f in report.findings:
                key = (f.ticker, f.metric, f.period, f.value, f.claim)
                if key in seen:
                    continue
                seen.add(key)
                n += 1
                existing.append(f.model_copy(update={"id": f"F{n}"}))
        return {"findings": existing}

    def writer(state: ResearchState) -> dict:
        lessons = mem.lessons()
        system = prompts.WRITER.format(
            lessons=("\nLessons from past reviews:\n" + "\n".join(f"- {x}" for x in lessons))
            if lessons
            else ""
        )
        user = f"Client question: {state['question']}\n\nFindings (JSON lines):\n{findings_digest(state['findings'])}"
        review: Review | None = state.get("review")
        if review and review.verdict != "approve" and state.get("memo"):
            issues = "\n".join(
                f"- [{i.kind}] {i.detail} {('| ' + i.excerpt) if i.excerpt else ''}"
                for i in review.issues
            )
            user += f"\n\nYour previous draft:\n{_strip_sources(state['memo'])}\n\nReviewer issues to fix:\n{issues}"
        if state.get("human_feedback"):
            user += f"\n\nFeedback from the human reviewer:\n{state['human_feedback']}"
        # A memo cut off at the token limit reads as unfinished to the critic and the client,
        # so a truncated draft is rewritten with more room instead of being passed on.
        for room in (4000, 8000):
            resp = llm.call(
                agent="writer",
                model=s.models.writer,
                system=system,
                messages=[{"role": "user", "content": user}],
                max_tokens=room,
            )
            if resp.get("stop_reason") != "max_tokens":
                break
        return {
            "memo": _strip_sources(text_of(resp)).strip()
            + "\n\n"
            + sources_section(state["findings"])
        }

    def critic(state: ResearchState) -> dict:
        review = review_memo(
            deps,
            state["question"],
            state["memo"],
            state["findings"],
            gap_prefix=f"G{state.get('research_rounds', 0) + 1}-",
        )
        mem.learn_from_review(review.issues, state["run_id"])
        return {"review": review}

    def route(state: ResearchState) -> str:
        r = state["review"]
        if (
            r.verdict == "research_more"
            and state.get("research_rounds", 0) < o.max_research_rounds - 1
        ):
            return "dispatch_gaps"
        if r.verdict in ("revise", "research_more") and state.get("revisions", 0) < o.max_revisions:
            return "count_revision"
        return "human_review" if o.human_review else "finalize"

    def count_revision(state: ResearchState) -> dict:
        return {"revisions": state.get("revisions", 0) + 1}

    def dispatch_gaps(state: ResearchState) -> dict:
        return {
            "tasks": state["review"].gap_tasks,
            "research_rounds": state.get("research_rounds", 0) + 1,
        }

    def human_review(state: ResearchState) -> dict:
        answer = interrupt({"memo": state["memo"], "review": state["review"].model_dump()})
        approved = (
            bool(answer.get("approve"))
            if isinstance(answer, dict)
            else str(answer).lower().startswith("y")
        )
        feedback = answer.get("feedback", "") if isinstance(answer, dict) else ""
        return {"approved": approved, "human_feedback": feedback}

    def after_human(state: ResearchState) -> str:
        return "finalize" if state.get("approved") or not state.get("human_feedback") else "writer"

    def finalize(state: ResearchState) -> dict:
        review = state.get("review")
        memo = state["memo"]
        if review and review.verdict != "approve":
            notes = "\n".join(f"- {i.detail}" for i in review.issues[:8])
            memo = memo.replace(
                "## Sources", f"## Reviewer notes (unresolved)\n{notes}\n\n## Sources", 1
            )
        verified = [f for f in state["findings"] if f.confidence != "low"]
        if review is None or review.verdict == "approve":
            mem.remember_facts(verified, state["run_id"])
        summary = _section(memo, "Summary") or memo[:400]
        tickers = sorted({f.ticker for f in state["findings"] if f.ticker})
        mem.remember_episode(
            state["run_id"],
            state["question"],
            summary,
            tickers,
            llm.tracer.totals()["cost_usd"],
            len(state["findings"]),
        )
        return {"memo": memo}

    # ------------------------------------------------------------------ wiring
    g = StateGraph(ResearchState)
    for name, fn in [
        ("recall", recall),
        ("planner", planner),
        ("worker", worker),
        ("merge", merge),
        ("writer", writer),
        ("critic", critic),
        ("count_revision", count_revision),
        ("dispatch_gaps", dispatch_gaps),
        ("human_review", human_review),
        ("finalize", finalize),
    ]:
        g.add_node(name, fn)
    g.add_edge(START, "recall")
    g.add_edge("recall", "planner")
    g.add_conditional_edges("planner", fan_out, ["worker", "merge"])
    g.add_edge("worker", "merge")
    g.add_edge("merge", "writer")
    g.add_edge("writer", "critic")
    g.add_conditional_edges(
        "critic", route, ["dispatch_gaps", "count_revision", "human_review", "finalize"]
    )
    g.add_edge("count_revision", "writer")
    g.add_conditional_edges("dispatch_gaps", fan_out, ["worker", "merge"])
    g.add_conditional_edges("human_review", after_human, ["finalize", "writer"])
    g.add_edge("finalize", END)
    return g.compile(checkpointer=checkpointer)


def _make_tasks(raw: list[dict], prefix: str, s: Settings) -> list[Task]:
    tasks = []
    for i, t in enumerate(raw[: s.orchestration.max_tasks], 1):
        if t.get("kind") not in ("quant", "filings") or not t.get("instruction"):
            continue
        tickers = [x.upper() for x in t.get("tickers", []) if x.upper() in s.universe.tickers]
        tasks.append(
            Task(
                id=f"{prefix}{i}",
                kind=t["kind"],
                instruction=t["instruction"],
                tickers=tickers,
                fiscal_years=t.get("fiscal_years", []),
            )
        )
    return tasks


def deterministic_issues(memo: str, findings: list[Finding]) -> list[Issue]:
    issues: list[Issue] = []
    # Numbers must be backed by a verified finding; low-confidence ones failed grounding checks.
    verified = [f for f in findings if f.confidence != "low"]
    for num in unsupported_numbers(_strip_sources(memo), verified):
        issues.append(
            Issue(
                kind="unsupported_number",
                detail=f"{num.raw} does not match any verified finding",
                excerpt=num.sentence[:200],
            )
        )
    uncited, unknown = citation_problems(memo, findings)
    issues += [
        Issue(kind="missing_citation", detail="Sentence with a number has no citation", excerpt=x)
        for x in uncited
    ]
    issues += [
        Issue(kind="unknown_citation", detail=f"Cites {c}, which is not a finding") for c in unknown
    ]
    return issues


def review_memo(
    deps: Deps,
    question: str,
    memo: str,
    findings: list[Finding],
    gap_prefix: str = "G1-",
    use_llm: bool = True,
) -> Review:
    """Critic: deterministic checks first, then an LLM review for what code can't judge."""
    auto = deterministic_issues(memo, findings)
    if not use_llm:
        return Review(verdict="revise" if auto else "approve", issues=auto)
    s, llm = deps.settings, deps.llm
    auto_text = "\n".join(f"- [{i.kind}] {i.detail} | {i.excerpt}" for i in auto) or "(none)"
    out = llm.structured(
        agent="critic",
        model=s.models.critic,
        system=prompts.CRITIC.format(auto_issues=auto_text),
        messages=[
            {
                "role": "user",
                "content": f"Client question: {question}\n\nFindings:\n"
                f"{findings_digest(findings)}\n\nDraft memo:\n{_strip_sources(memo)}",
            }
        ],
        schema=REVIEW_SCHEMA,
        max_tokens=1500,
    )
    llm_issues = [
        Issue(**i)
        for i in out.get("issues", [])
        if i.get("kind") in ("unsupported_claim", "missing_evidence", "other")
    ]
    verdict = out.get("verdict", "revise")
    if auto and verdict == "approve":
        verdict = "revise"  # deterministic failures always block approval
    gaps = (
        _make_tasks(out.get("gap_tasks") or [], gap_prefix, s) if verdict == "research_more" else []
    )
    if verdict == "research_more" and not gaps:
        verdict = "revise"
    return Review(verdict=verdict, issues=auto + llm_issues, gap_tasks=gaps)


def sources_section(findings: list[Finding]) -> str:
    lines = ["## Sources"]
    for f in findings:
        flag = " *(low confidence)*" if f.confidence == "low" else ""
        quote = f' — "{_clip(f.source.quote, 200)}"' if f.source.quote else ""
        lines.append(f"- **[{f.id}]** `{f.source.ref}`{quote}{flag}")
    return "\n".join(lines)


def _clip(text: str, n: int) -> str:
    """Shorten at a word boundary, marking the cut."""
    text = " ".join(text.split())
    return text if len(text) <= n else text[:n].rsplit(" ", 1)[0] + " …"


def _strip_sources(memo: str) -> str:
    return re.split(r"\n## (?:Sources|Reviewer notes)", memo)[0]


def _section(memo: str, title: str) -> str:
    m = re.search(rf"## {title}\s*\n(.*?)(?=\n## |\Z)", memo, re.DOTALL)
    return m.group(1).strip() if m else ""


def new_run_id() -> str:
    return uuid.uuid4().hex[:10]
