import json

from equity_agents.evals.fault_injection import deterministic_detects, mutate
from equity_agents.graph import build_graph, review_memo
from equity_agents.llm import compact_history
from equity_agents.memory.store import MemoryStore
from equity_agents.runner import research, resume
from equity_agents.state import Finding, Issue, Source
from equity_agents.tracing import load_trace

QUESTION = "How did NVIDIA's revenue change in fiscal 2025 and why?"


def test_happy_path_end_to_end(settings, make):
    deps, backend = make()
    out = research(QUESTION, settings, run_id="r1", deps=deps)

    assert "$130.5 billion" in out.memo and "114.2%" in out.memo
    assert "## Sources" in out.memo and "xbrl:NVDA:revenue:FY2025" in out.memo
    summary = json.loads((out.run_dir / "summary.json").read_text())
    assert summary["verdict"] == "approve" and summary["revisions"] == 0
    assert {t["kind"] for t in summary["plan"]} == {"quant", "filings"}
    assert summary["low_confidence_findings"] == 0
    trace = load_trace(out.run_dir / "trace.jsonl")
    assert {e["name"] for e in trace if e["kind"] == "tool"} == {
        "get_metric",
        "calculate",
        "search_filings",
    }
    assert summary["cost_usd"] > 0 and set(summary["by_agent"]) >= {"planner", "writer", "critic"}
    # Prompt caching breakpoints on the static prefix of every call.
    assert all(r["system"][0]["cache_control"] == {"type": "ephemeral"} for r in backend.requests)


def test_critic_forces_revision_of_wrong_number(settings, make):
    deps, backend = make(first_draft_wrong=True)
    out = research(QUESTION, settings, run_id="r2", deps=deps)
    assert backend.writer_calls == 2
    assert "$131.9 billion" not in out.memo and "$130.5 billion" in out.memo
    assert out.state["revisions"] == 1 and out.state["review"].verdict == "approve"
    # The revision request carried the critic's issue back to the writer.
    second_writer = [
        r for r in backend.requests if "writer on an equity" in r["system"][0]["text"]
    ][1]
    assert "unsupported_number" in second_writer["messages"][0]["content"]
    # Procedural memory learned from the review.
    assert any("Every number" in lesson for lesson in deps.memory.lessons())


def test_grounding_check_catches_wrong_tool_value(settings, make):
    deps, _ = make(quant_lies=True)
    out = research(QUESTION, settings, run_id="r3", deps=deps)
    bad = [f for f in out.state["findings"] if f.confidence == "low"]
    assert bad and "does not match XBRL" in bad[0].claim
    # The writer used the bad number, the critic blocked it, revisions ran out.
    assert out.state["review"].verdict == "revise"
    assert "## Reviewer notes (unresolved)" in out.memo
    assert deps.memory.stats()["facts"] == 0  # unapproved runs don't write semantic memory


def test_memory_is_written_and_recalled(settings, make):
    deps, _ = make(run_id="m1")
    research(QUESTION, settings, run_id="m1", deps=deps)
    mem = deps.memory
    assert mem.stats()["facts"] == 3 and mem.stats()["episodes"] == 1  # 2 xbrl + 1 filing fact
    deps2, backend2 = make(run_id="m2", memory=mem)
    out = research(
        "What was NVIDIA's revenue growth in fiscal 2025?", settings, run_id="m2", deps=deps2
    )
    assert (
        "Facts in memory" in out.state["memory_context"]
        and "Related past research" in out.state["memory_context"]
    )
    planner_req = backend2.requests[0]
    assert "xbrl:NVDA:revenue:FY2025" in planner_req["messages"][0]["content"]
    # Re-confirming the same facts increments confidence instead of duplicating them.
    assert mem.stats()["facts"] == 3


def test_semantic_memory_versions_conflicts(tmp_path):
    mem = MemoryStore(tmp_path / "m.sqlite")
    f = Finding(
        claim="rev",
        ticker="NVDA",
        metric="revenue",
        period="FY2025",
        value=1.0,
        unit="USD",
        source=Source(kind="xbrl", ref="a"),
    )
    assert mem.remember_facts([f], "r1") == {"new": 1}
    assert mem.remember_facts([f], "r2") == {"confirmed": 1}
    assert mem.remember_facts([f.model_copy(update={"value": 2.0})], "r3") == {
        "conflicts": 1,
        "new": 1,
    }
    assert mem.stats()["superseded_facts"] == 1 and mem.recall_facts(["NVDA"])[0]["value"] == 2.0


def test_human_in_the_loop_pause_and_resume(settings, make):
    s = settings.model_copy(
        update={"orchestration": settings.orchestration.model_copy(update={"human_review": True})}
    )
    deps, _ = make(run_id="h1", settings_override=s)
    out = research(QUESTION, s, run_id="h1", deps=deps)
    assert out.interrupted and "memo" in out.state["__interrupt__"][0].value
    deps2, _ = make(run_id="h1", memory=deps.memory, settings_override=s)
    final = resume("h1", s, approve=True, deps=deps2)
    assert not final.interrupted and final.state["approved"] is True


def test_compaction_masks_old_tool_results():
    msgs = [{"role": "user", "content": "task"}]
    for i in range(5):
        msgs.append(
            {
                "role": "assistant",
                "content": [{"type": "tool_use", "id": f"t{i}", "name": "x", "input": {}}],
            }
        )
        msgs.append(
            {
                "role": "user",
                "content": [{"type": "tool_result", "tool_use_id": f"t{i}", "content": "y" * 500}],
            }
        )
    assert compact_history(msgs, keep_recent=2) == 3
    contents = [
        m["content"][0]["content"]
        for m in msgs
        if m["role"] == "user" and isinstance(m["content"], list)
    ]
    assert all(c.startswith("[compacted") for c in contents[:3]) and contents[3] == "y" * 500


def test_fault_injection_detection(settings, make):
    deps, _ = make(run_id="f1")
    out = research(QUESTION, settings, run_id="f1", deps=deps)
    findings = out.state["findings"]
    import random

    rng = random.Random(0)
    assert not deterministic_detects(out.memo, findings)  # clean memo passes
    for kind in ("perturb_number", "fabricate_number", "drop_citation", "unknown_citation"):
        corrupted = mutate(out.memo, kind, findings, rng, ["NVIDIA", "Microsoft"])
        assert corrupted and deterministic_detects(corrupted, findings), kind
    review = review_memo(
        deps,
        QUESTION,
        mutate(out.memo, "perturb_number", findings, rng, []),
        findings,
        use_llm=False,
    )
    assert review.verdict == "revise" and review.issues[0].kind == "unsupported_number"


def test_graph_topology(settings, make):
    deps, _ = make()
    nodes = set(build_graph(deps).get_graph().nodes)
    assert {
        "recall",
        "planner",
        "worker",
        "merge",
        "writer",
        "critic",
        "human_review",
        "finalize",
    } <= nodes


def test_issue_kinds_map_to_lessons(tmp_path):
    mem = MemoryStore(tmp_path / "m.sqlite")
    mem.learn_from_review(
        [Issue(kind="missing_citation", detail="x"), Issue(kind="missing_citation", detail="y")],
        "r",
    )
    mem.learn_from_review([Issue(kind="missing_citation", detail="z")], "r2")
    assert len(mem.lessons()) == 1


def test_memory_is_thread_safe(tmp_path):
    """Parallel workers read memory concurrently while other nodes write it."""
    from concurrent.futures import ThreadPoolExecutor

    mem = MemoryStore(tmp_path / "m.sqlite")
    f = Finding(
        claim="rev",
        ticker="NVDA",
        metric="revenue",
        period="FY2025",
        value=1.0,
        unit="USD",
        source=Source(kind="xbrl", ref="a"),
    )

    def work(i):
        if i % 10 == 0:
            mem.remember_facts([f.model_copy(update={"period": f"FY{2000 + i}"})], f"r{i}")
            mem.remember_episode(f"r{i}", "NVIDIA revenue", "summary", ["NVDA"], 0.0, 1)
        return (
            len(mem.recall_facts(["NVDA"])),
            mem.similar_episodes("NVIDIA revenue"),
            mem.lessons(),
            mem.stats(),
        )

    with ThreadPoolExecutor(max_workers=32) as pool:
        list(pool.map(work, range(1000)))
    assert mem.stats()["facts"] == 100


def test_writer_rewrites_a_memo_cut_off_at_the_token_limit(settings, make):
    deps, backend = make()
    original = backend.create
    limits = []

    def create(**req):
        resp = original(**req)
        if "writer on an equity" in req["system"][0]["text"]:
            limits.append(req["max_tokens"])
            if len(limits) == 1:
                cut = resp["content"][0]["text"][:120]
                return {
                    **resp,
                    "content": [{"type": "text", "text": cut}],
                    "stop_reason": "max_tokens",
                }
        return resp

    backend.create = create
    out = research(QUESTION, settings, run_id="cut", deps=deps)
    assert limits[:2] == [4000, 8000] and "## Risks and caveats" in out.memo
