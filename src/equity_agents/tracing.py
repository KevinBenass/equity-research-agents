"""Lightweight tracing: every LLM call and tool call becomes a span in runs/<id>/trace.jsonl.

Enough to answer the questions that matter when operating an agent system: which agent
spent the tokens, which tool was slow, how many steps a worker took, where a run failed.
No external service required; the JSONL maps directly onto OpenTelemetry spans if needed.
"""

from __future__ import annotations

import json
import threading
import time
import uuid
from collections import defaultdict
from contextlib import contextmanager
from contextvars import ContextVar
from pathlib import Path
from typing import Any

_current_span: ContextVar[str | None] = ContextVar("current_span", default=None)


class Tracer:
    def __init__(self, run_id: str, path: Path | None = None) -> None:
        self.run_id = run_id
        self.path = path
        self.events: list[dict[str, Any]] = []
        self._lock = threading.Lock()
        if path:
            path.parent.mkdir(parents=True, exist_ok=True)

    @contextmanager
    def span(self, name: str, kind: str, **attrs: Any):
        span_id = uuid.uuid4().hex[:12]
        parent = _current_span.get()
        token = _current_span.set(span_id)
        start = time.perf_counter()
        record: dict[str, Any] = {
            "span_id": span_id,
            "parent": parent,
            "name": name,
            "kind": kind,
            **attrs,
        }
        try:
            yield record
            record["status"] = "ok"
        except Exception as exc:
            record["status"] = "error"
            record["error"] = f"{type(exc).__name__}: {exc}"
            raise
        finally:
            record["ms"] = round((time.perf_counter() - start) * 1000, 1)
            _current_span.reset(token)
            self._emit(record)

    def _emit(self, record: dict[str, Any]) -> None:
        with self._lock:
            self.events.append(record)
            if self.path:
                with self.path.open("a") as f:
                    f.write(json.dumps(record, default=str) + "\n")

    # ---------------------------------------------------------------- summaries
    def totals(self) -> dict[str, Any]:
        llm = [e for e in self.events if e["kind"] == "llm"]
        tools = [e for e in self.events if e["kind"] == "tool"]
        by_agent: dict[str, dict[str, float]] = defaultdict(lambda: defaultdict(float))
        for e in llm:
            a = by_agent[e.get("agent", "?")]
            a["calls"] += 1
            a["input_tokens"] += e.get("input_tokens", 0)
            a["output_tokens"] += e.get("output_tokens", 0)
            a["cost_usd"] += e.get("cost_usd", 0.0)
        tool_counts: dict[str, int] = defaultdict(int)
        for e in tools:
            tool_counts[e["name"]] += 1
        return {
            "llm_calls": len(llm),
            "tool_calls": len(tools),
            "tool_errors": sum(
                e.get("status") == "error" or bool(e.get("is_error")) for e in tools
            ),
            "input_tokens": int(sum(e.get("input_tokens", 0) for e in llm)),
            "cache_read_tokens": int(sum(e.get("cache_read_tokens", 0) for e in llm)),
            "output_tokens": int(sum(e.get("output_tokens", 0) for e in llm)),
            "cost_usd": round(sum(e.get("cost_usd", 0.0) for e in llm), 4),
            "by_agent": {k: dict(v) for k, v in by_agent.items()},
            "tools": dict(tool_counts),
        }


def load_trace(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
