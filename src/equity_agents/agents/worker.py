"""The researcher loop shared by the quant and filings specialists.

Context engineering lives here:
- The worker starts from a self-contained task brief plus only the memory facts about its
  own tickers, never the full conversation of the team.
- Tool outputs are truncated at the source and, once the worker's history exceeds its token
  budget, older tool results are compacted to one-line previews (observation masking).
- The output is structured (``submit_findings``), so the handoff is data, not prose.

Grounding checks run in code after the model submits:
- A finding's source ref must have appeared in a tool result during this loop.
- XBRL findings are re-checked against the database; calc findings are re-evaluated.
Findings failing a check are kept but downgraded to low confidence and annotated, so the
writer and critic can see them for what they are.
"""

from __future__ import annotations

import json
import re
from typing import Any

from equity_agents.llm import LLM, compact_history, estimate_tokens, strip_private
from equity_agents.state import Finding, Task, WorkerReport
from equity_agents.tools.base import Tool, run_tool
from equity_agents.tools.calc import safe_eval
from equity_agents.tools.xbrl import FinancialsDB

FINDING_SCHEMA = {
    "type": "object",
    "properties": {
        "findings": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "claim": {"type": "string"},
                    "ticker": {"type": "string"},
                    "metric": {"type": "string"},
                    "value": {"type": "number"},
                    "unit": {"type": "string"},
                    "period": {"type": "string", "description": "e.g. FY2025"},
                    "source": {
                        "type": "object",
                        "properties": {
                            "kind": {
                                "type": "string",
                                "enum": ["xbrl", "filing", "calc", "memory"],
                            },
                            "ref": {"type": "string"},
                            "quote": {"type": "string"},
                        },
                        "required": ["kind", "ref"],
                    },
                    "confidence": {"type": "string", "enum": ["high", "medium", "low"]},
                },
                "required": ["claim", "source"],
            },
        },
        "notes": {
            "type": "string",
            "description": "Anything the lead analyst should know (gaps, caveats).",
        },
    },
    "required": ["findings"],
}
SUBMIT = {
    "name": "submit_findings",
    "description": "Return your findings. Call once, at the end.",
    "input_schema": FINDING_SCHEMA,
}
REF_RE = re.compile(r"\b((?:xbrl|filing):[\w.\-:]+)")


def brief(task: Task, facts: list[dict]) -> str:
    lines = [f"Task {task.id}: {task.instruction}"]
    if task.tickers:
        lines.append(f"Tickers: {', '.join(task.tickers)}")
    if task.fiscal_years:
        lines.append(f"Fiscal years: {', '.join(map(str, task.fiscal_years))}")
    if facts:
        lines.append(
            "\nKnown facts from memory (verify with tools before relying on them; cite their ref if confirmed):"
        )
        lines += [
            f"- {f['ticker']} {f['metric'] or ''} {f['period'] or ''}: {f['value'] if f['value'] is not None else f['claim']} ({f['ref']})"
            for f in facts
        ]
    return "\n".join(lines)


def verify(findings: list[Finding], seen_refs: set[str], db: FinancialsDB | None) -> list[Finding]:
    for f in findings:
        problems = []
        ref = f.source.ref
        if f.source.kind in ("xbrl", "filing") and ref not in seen_refs:
            problems.append("source ref never returned by a tool")
        if f.source.kind == "xbrl" and db is not None and f.value is not None:
            m = re.match(r"xbrl:(\w+):(\w+):FY(\d{4})", ref)
            truth = db.get(m.group(1), m.group(2), int(m.group(3))) if m else None
            if truth is None or abs(truth - f.value) > 1e-6 * max(1.0, abs(truth)):
                problems.append(f"value does not match XBRL ({truth})")
        if f.source.kind == "calc" and f.value is not None:
            try:
                expected = safe_eval(ref.removeprefix("calc:"))
                if abs(expected - f.value) > 0.051 + 1e-4 * abs(expected):
                    problems.append(f"calc result is {expected:.6g}, not {f.value}")
            except Exception:
                problems.append("calc expression does not evaluate")
        if problems:
            f.confidence = "low"
            f.claim = f"{f.claim} [UNVERIFIED: {'; '.join(problems)}]"
    return findings


def run_worker(
    task: Task,
    *,
    llm: LLM,
    model: str,
    system: str,
    tools: list[Tool],
    facts: list[dict],
    max_steps: int,
    budget_tokens: int,
    keep_recent: int,
    max_chars: int,
    db: FinancialsDB | None = None,
) -> WorkerReport:
    agent = f"{task.kind}_worker[{task.id}]"
    by_name = {t.name: t for t in tools}
    specs = [t.spec() for t in tools] + [SUBMIT]
    messages: list[dict[str, Any]] = [{"role": "user", "content": brief(task, facts)}]
    seen_refs: set[str] = {f["ref"] for f in facts}
    called: list[str] = []
    compacted = 0

    for step in range(1, max_steps + 1):
        force = step == max_steps
        resp = llm.call(
            agent=agent,
            model=model,
            system=system,
            messages=strip_private(messages),
            tools=specs,
            tool_choice={"type": "tool", "name": "submit_findings"} if force else None,
        )
        messages.append({"role": "assistant", "content": resp["content"]})
        uses = [b for b in resp["content"] if b["type"] == "tool_use"]
        if not uses:
            messages.append(
                {"role": "user", "content": "Call submit_findings with your findings now."}
            )
            continue
        results = []
        for use in uses:
            if use["name"] == "submit_findings":
                raw = use["input"].get("findings", [])
                findings = []
                for item in raw:
                    try:
                        findings.append(Finding.model_validate({**item, "task_id": task.id}))
                    except ValueError:
                        continue
                notes = use["input"].get("notes", "")
                if compacted:
                    notes = f"{notes} (context compacted {compacted}x)".strip()
                return WorkerReport(
                    task_id=task.id,
                    findings=verify(findings, seen_refs, db),
                    steps=step,
                    tool_calls=called,
                    notes=notes,
                )
            called.append(use["name"])
            tool = by_name.get(use["name"])
            if tool is None:
                text, is_error = f"Error: unknown tool {use['name']}", True
            else:
                text, is_error = run_tool(tool, use["input"], llm.tracer, agent, max_chars)
            seen_refs |= set(REF_RE.findall(text))
            results.append(
                {
                    "type": "tool_result",
                    "tool_use_id": use["id"],
                    "content": text,
                    "is_error": is_error,
                }
            )
        messages.append({"role": "user", "content": results})
        if estimate_tokens(messages) > budget_tokens:
            compacted += compact_history(messages, keep_recent)

    return WorkerReport(
        task_id=task.id,
        findings=[],
        steps=max_steps,
        tool_calls=called,
        notes="Worker hit the step limit without submitting findings.",
    )


def findings_digest(findings: list[Finding]) -> str:
    """Compact JSON lines for the writer and critic: only what they need."""
    rows = []
    for f in findings:
        row = {"id": f.id, "claim": f.claim, "confidence": f.confidence, "source": f.source.ref}
        if f.value is not None:
            row.update(value=f.value, unit=f.unit, period=f.period, ticker=f.ticker)
        if f.source.quote:
            row["quote"] = f.source.quote[:240]
        rows.append(json.dumps(row))
    return "\n".join(rows)
