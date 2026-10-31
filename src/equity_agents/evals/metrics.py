"""Scoring a run: is the answer right, is it grounded, did the team work sensibly, at what cost?"""

from __future__ import annotations

from typing import Any

from equity_agents.evals.dataset import EvalItem, TruthNumber
from equity_agents.graph import _strip_sources
from equity_agents.numbers import citation_problems, extract_numbers, unsupported_numbers
from equity_agents.state import Finding

QUANT_TOOLS = {"get_metric", "query_financials", "recall_facts"}
FILINGS_TOOLS = {"search_filings"}


def truth_found(memo: str, t: TruthNumber) -> bool:
    for n in extract_numbers(memo):
        if t.unit == "%" and n.kind == "pct":
            # Allow rounding to the displayed precision, plus 0.1pp for rounding of inputs.
            if abs(abs(n.value) - abs(t.value)) <= n.tolerance + 0.1:
                return True
        elif t.unit == "USD" and n.kind in ("money", "plain"):
            if abs(n.value - t.value) <= n.tolerance + 1e-6 * abs(t.value):
                return True
    return False


def score_run(
    item: EvalItem, memo: str, findings: list[Finding], summary: dict[str, Any], trace: list[dict]
) -> dict[str, Any]:
    body = _strip_sources(memo)
    found = [truth_found(body, t) for t in item.truth]
    numbers = extract_numbers(body)
    unsupported = unsupported_numbers(body, findings)
    uncited, unknown = citation_problems(memo, findings)
    numbered_sentences = {n.sentence for n in numbers}
    tools_used = {e["name"] for e in trace if e["kind"] == "tool"}
    kinds_used = {
        k for k, names in (("quant", QUANT_TOOLS), ("filings", FILINGS_TOOLS)) if tools_used & names
    }
    return {
        "qid": item.qid,
        "template": item.template,
        "numeric_recall": sum(found) / len(found) if found else None,
        "all_numbers_correct": all(found),
        "n_numbers": len(numbers),
        "unsupported_numbers": len(unsupported),
        "citation_coverage": 1 - len(uncited) / len(numbered_sentences)
        if numbered_sentences
        else 1.0,
        "unknown_citations": len(unknown),
        "required_specialists_used": set(item.requires) <= kinds_used,
        "verdict": summary.get("verdict"),
        "revisions": summary.get("revisions", 0),
        "research_rounds": summary.get("research_rounds", 0),
        "n_tasks": len(summary.get("plan", [])),
        "llm_calls": summary.get("llm_calls", 0),
        "tool_calls": summary.get("tool_calls", 0),
        "tool_errors": summary.get("tool_errors", 0),
        "cost_usd": summary.get("cost_usd", 0.0),
        "latency_s": summary.get("latency_s", 0.0),
        "low_confidence_findings": summary.get("low_confidence_findings", 0),
    }
