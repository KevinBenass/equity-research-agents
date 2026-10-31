"""Tool definition and execution with tracing and output truncation at the source."""

from __future__ import annotations

import json
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from equity_agents.tracing import Tracer


class ToolError(Exception):
    """An error the agent should see and can recover from (bad SQL, unknown ticker...)."""


@dataclass
class Tool:
    name: str
    description: str
    input_schema: dict[str, Any]
    fn: Callable[..., Any]

    def spec(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "description": self.description,
            "input_schema": self.input_schema,
        }


def run_tool(
    tool: Tool, args: dict[str, Any], tracer: Tracer, agent: str, max_chars: int
) -> tuple[str, bool]:
    """Execute a tool; return (text for the model, is_error). Never raises on ToolError."""
    with tracer.span(tool.name, "tool", agent=agent, args=args) as rec:
        try:
            out = tool.fn(**args)
            text = out if isinstance(out, str) else json.dumps(out, default=str)
            is_error = False
        except (ToolError, TypeError, ValueError) as exc:
            text, is_error = f"Error: {exc}", True
        if len(text) > max_chars:
            text = (
                text[:max_chars] + f"\n…[truncated {len(text) - max_chars} chars; narrow the query]"
            )
        rec.update(is_error=is_error, result_chars=len(text))
    return text, is_error
