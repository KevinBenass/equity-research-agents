"""Checks that every request the agents build is one the real Claude API accepts.

The scripted backend in fakes.py enforces message-level rules on every call; these tests
cover the request schema itself, so a paid run never fails on a 400.
"""

import inspect

import anthropic

from equity_agents.agents.worker import FINDING_SCHEMA
from equity_agents.graph import PLAN_SCHEMA, REVIEW_SCHEMA
from equity_agents.llm import LLM
from equity_agents.runner import research
from equity_agents.tracing import Tracer

UNSUPPORTED = {
    "minimum",
    "maximum",
    "exclusiveMinimum",
    "exclusiveMaximum",
    "multipleOf",
    "minLength",
    "maxLength",
    "maxItems",
    "uniqueItems",
    "patternProperties",
}


def _walk(node, path="$"):
    if isinstance(node, dict):
        assert not UNSUPPORTED & set(node), f"{path}: {UNSUPPORTED & set(node)}"
        if node.get("type") == "object":
            assert node.get("additionalProperties") is False, f"{path}: additionalProperties"
            assert set(node["required"]) == set(node["properties"]), f"{path}: all required"
        if "minItems" in node:
            assert node["minItems"] in (0, 1), path
        for k, v in node.items():
            _walk(v, f"{path}.{k}")
    elif isinstance(node, list):
        for i, v in enumerate(node):
            _walk(v, f"{path}[{i}]")


def test_structured_output_schemas_respect_api_limits():
    for schema in (PLAN_SCHEMA, REVIEW_SCHEMA, FINDING_SCHEMA):
        _walk(schema)


def test_requests_match_installed_sdk_signature(settings, make):
    accepted = set(inspect.signature(anthropic.resources.messages.Messages.create).parameters)
    deps, backend = make()
    research(
        "How did NVIDIA's revenue change in fiscal 2025 and why?", settings, run_id="sig", deps=deps
    )
    assert backend.requests
    for request in backend.requests:
        assert set(request) <= accepted, set(request) - accepted
    assert not any(
        (r.get("tool_choice") or {}).get("type") in ("tool", "any") for r in backend.requests
    )


def test_structured_falls_back_when_tool_choice_none_is_rejected(settings, monkeypatch):
    class Picky:
        def __init__(self):
            self.requests = []

        def create(self, **req):
            self.requests.append(req)
            if "tool_choice" in req:
                raise RuntimeError("tool_choice: type none is not supported for this model")
            return {
                "content": [{"type": "text", "text": '{"ok": true}'}],
                "stop_reason": "end_turn",
                "usage": {"input_tokens": 1, "output_tokens": 1},
            }

    monkeypatch.setattr(LLM, "_tool_choice_none_unsupported", False)
    backend = Picky()
    llm = LLM(settings.models.prices, Tracer("t"), backend)
    tool = {"name": "x", "description": "x", "input_schema": {"type": "object", "properties": {}}}
    out = llm.structured(
        agent="a",
        model="m",
        system="s",
        messages=[{"role": "user", "content": "q"}],
        schema={"type": "object"},
        tools=[tool],
    )
    assert out == {"ok": True} and "tool_choice" not in backend.requests[-1]


def test_structured_retries_when_cut_off(settings):
    calls = []

    class Short:
        def create(self, **req):
            calls.append(req["max_tokens"])
            stop = "max_tokens" if len(calls) == 1 else "end_turn"
            return {
                "content": [{"type": "text", "text": '{"a"' if len(calls) == 1 else '{"a": 1}'}],
                "stop_reason": stop,
                "usage": {"input_tokens": 1, "output_tokens": 1},
            }

    llm = LLM(settings.models.prices, Tracer("t"), Short())
    out = llm.structured(
        agent="a",
        model="m",
        system="s",
        messages=[{"role": "user", "content": "q"}],
        schema={"type": "object"},
        max_tokens=100,
    )
    assert out == {"a": 1} and calls == [100, 300]
