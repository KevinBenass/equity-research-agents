"""Claude client used by every agent: prompt caching, cost accounting, tracing, and a
generic tool-use loop with context compaction.

Responses are normalized to plain dicts so agents (and the scripted fake used in tests)
don't depend on SDK object shapes.
"""

from __future__ import annotations

import json
import re
from collections.abc import Callable
from typing import Any, Protocol

from equity_agents.config import Price
from equity_agents.tracing import Tracer


class Backend(Protocol):
    def create(self, **request: Any) -> dict[str, Any]: ...


class AnthropicBackend:
    def __init__(self) -> None:
        import anthropic

        self._client = anthropic.Anthropic(max_retries=5)

    def create(self, **request: Any) -> dict[str, Any]:
        resp = self._client.messages.create(**request)
        content = []
        for b in resp.content:
            if b.type == "text":
                content.append({"type": "text", "text": b.text})
            elif b.type == "tool_use":
                content.append({"type": "tool_use", "id": b.id, "name": b.name, "input": b.input})
        u = resp.usage
        return {
            "content": content,
            "stop_reason": resp.stop_reason,
            "usage": {
                "input_tokens": u.input_tokens,
                "output_tokens": u.output_tokens,
                "cache_read_tokens": getattr(u, "cache_read_input_tokens", 0) or 0,
                "cache_write_tokens": getattr(u, "cache_creation_input_tokens", 0) or 0,
            },
        }


class LLM:
    def __init__(
        self, prices: dict[str, Price], tracer: Tracer, backend: Backend | None = None
    ) -> None:
        self.prices = prices
        self.tracer = tracer
        self._backend = backend

    @property
    def backend(self) -> Backend:
        if self._backend is None:
            self._backend = AnthropicBackend()
        return self._backend

    def cost(self, model: str, usage: dict[str, int]) -> float:
        p = self.prices.get(model)
        if not p:
            return 0.0
        return (
            usage.get("input_tokens", 0) * p.input
            + usage.get("cache_write_tokens", 0) * p.input * 1.25
            + usage.get("cache_read_tokens", 0) * p.input * 0.10
            + usage.get("output_tokens", 0) * p.output
        ) / 1e6

    def call(
        self,
        *,
        agent: str,
        model: str,
        system: str,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
        tool_choice: dict[str, Any] | None = None,
        max_tokens: int = 2048,
    ) -> dict[str, Any]:
        # Cache breakpoints on the static prefix (system prompt and tool definitions),
        # which every step of a tool loop repeats.
        request: dict[str, Any] = {
            "model": model,
            "max_tokens": max_tokens,
            "system": [{"type": "text", "text": system, "cache_control": {"type": "ephemeral"}}],
            "messages": messages,
        }
        if tools:
            tools = [dict(t) for t in tools]
            tools[-1] = {**tools[-1], "cache_control": {"type": "ephemeral"}}
            request["tools"] = tools
        if tool_choice:
            request["tool_choice"] = tool_choice
        with self.tracer.span(f"{agent}.llm", "llm", agent=agent, model=model) as rec:
            resp = self.backend.create(**request)
            u = resp["usage"]
            rec.update(
                input_tokens=u.get("input_tokens", 0),
                output_tokens=u.get("output_tokens", 0),
                cache_read_tokens=u.get("cache_read_tokens", 0),
                cost_usd=self.cost(model, u),
                stop_reason=resp.get("stop_reason"),
            )
        return resp


def text_of(resp: dict[str, Any]) -> str:
    return "".join(b["text"] for b in resp["content"] if b["type"] == "text")


def parse_json(text: str) -> Any:
    fenced = re.search(r"```(?:json)?\s*(.*?)```", text, re.DOTALL)
    candidate = fenced.group(1) if fenced else text
    starts = [i for i in (candidate.find("{"), candidate.find("[")) if i != -1]
    if not starts:
        raise ValueError(f"No JSON in output: {text[:200]!r}")
    obj, _ = json.JSONDecoder().raw_decode(candidate[min(starts) :])
    return obj


def estimate_tokens(obj: Any) -> int:
    return len(json.dumps(obj, default=str)) // 4


def compact_history(messages: list[dict[str, Any]], keep_recent: int) -> int:
    """Observation masking: replace all but the most recent tool results with a stub.

    The model keeps its own reasoning and the tool calls it made (so it knows what it
    already tried), but old raw tool outputs, the bulk of a worker's context, are cut to
    a short preview. Returns the number of results compacted.
    """
    positions = [
        (i, j)
        for i, m in enumerate(messages)
        if m["role"] == "user" and isinstance(m["content"], list)
        for j, block in enumerate(m["content"])
        if block.get("type") == "tool_result" and not block.get("_compacted")
    ]
    n = 0
    for i, j in positions[: max(0, len(positions) - keep_recent)]:
        block = messages[i]["content"][j]
        raw = (
            block["content"] if isinstance(block["content"], str) else json.dumps(block["content"])
        )
        block["content"] = f"[compacted earlier result; preview: {raw[:160]}…]"
        block["_compacted"] = True
        n += 1
    return n


def strip_private(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Remove bookkeeping keys (leading underscore) before sending to the API."""
    out = []
    for m in messages:
        if isinstance(m["content"], list):
            out.append(
                {
                    **m,
                    "content": [
                        {k: v for k, v in b.items() if not k.startswith("_")} for b in m["content"]
                    ],
                }
            )
        else:
            out.append(m)
    return out


ToolFn = Callable[[dict[str, Any]], str]
