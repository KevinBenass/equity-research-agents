from fakes import PASSAGES, ScriptedBackend, _json, _text, check_request

from equity_agents.agents.worker import run_worker
from equity_agents.evals.dataset import EvalItem, TruthNumber, build_eval_set
from equity_agents.evals.metrics import truth_found
from equity_agents.evals.run import fault_injection, report, run_items, summarize
from equity_agents.llm import LLM
from equity_agents.memory.store import MemoryStore
from equity_agents.state import Task
from equity_agents.tools.filings import static_filings_tool
from equity_agents.tracing import Tracer


def test_eval_set_truth_comes_from_xbrl(db):
    items = build_eval_set(db, [2023, 2024, 2025], seed=1)
    templates = {i.template for i in items}
    assert {"growth", "margin", "comparison", "drivers"} <= templates
    nvda_margin = (
        next(i for i in items if i.qid == "margin-NVDA-2025")
        if any(i.qid == "margin-NVDA-2025" for i in items)
        else None
    )
    if nvda_margin:
        assert abs(nvda_margin.truth[0].value - 81_453 / 130_497 * 100) < 1e-6
    assert all("Nvidia" in i.question or "Microsoft" in i.question for i in items)


def test_truth_matching_respects_display_precision():
    assert truth_found(
        "Revenue was $130.5 billion.", TruthNumber(label="r", value=130_497_000_000, unit="USD")
    )
    assert not truth_found(
        "Revenue was $131.5 billion.", TruthNumber(label="r", value=130_497_000_000, unit="USD")
    )
    assert truth_found("up 114.2%", TruthNumber(label="g", value=114.2049, unit="%"))
    assert truth_found("margin of 62%", TruthNumber(label="m", value=62.418, unit="%"))


def test_worker_downgrades_findings_with_unseen_refs(settings, db):
    class Liar(ScriptedBackend):
        """Reports a wrong XBRL value and cites a filing passage no tool ever returned."""

        def create(self, **req):
            self.requests.append(req)
            check_request(req)
            if "output_config" not in req:
                return _text("Done.")  # no research at all
            return _json(
                {
                    "findings": [
                        {
                            "claim": "Revenue",
                            "ticker": "NVDA",
                            "metric": "revenue",
                            "value": 1.0,
                            "unit": "USD",
                            "period": "FY2025",
                            "source": {
                                "kind": "xbrl",
                                "ref": "xbrl:NVDA:revenue:FY2025",
                                "quote": "",
                            },
                            "confidence": "high",
                        },
                        {
                            "claim": "Demand was strong",
                            "ticker": "NVDA",
                            "metric": "",
                            "value": None,
                            "unit": "",
                            "period": "",
                            "source": {
                                "kind": "filing",
                                "ref": "filing:NVDA-10-K-FY2025::x::9",
                                "quote": "",
                            },
                            "confidence": "high",
                        },
                    ],
                    "notes": "",
                }
            )

    llm = LLM(settings.models.prices, Tracer("w"), Liar())
    report_ = run_worker(
        Task(id="T1", kind="quant", instruction="x"),
        llm=llm,
        model="m",
        system="s",
        tools=db.tools(),
        facts=[],
        max_steps=3,
        budget_tokens=10_000,
        keep_recent=2,
        max_chars=2000,
        db=db,
    )
    xbrl, filing = report_.findings
    assert xbrl.confidence == "low" and "does not match XBRL" in xbrl.claim
    assert filing.confidence == "low" and "never returned by a tool" in filing.claim
    assert filing.metric is None and filing.unit is None  # "" mapped back to None


def test_eval_runner_end_to_end(settings, db, tmp_path):
    item = EvalItem(
        qid="growth-NVDA-revenue-2025",
        template="growth",
        tickers=["NVDA"],
        requires=["quant"],
        question="How did NVIDIA's revenue change in fiscal 2025 and why?",
        truth=[
            TruthNumber(label="rev", value=130_497_000_000, unit="USD"),
            TruthNumber(label="growth", value=114.2049, unit="%"),
        ],
    )
    kwargs = {"backend": ScriptedBackend(), "filings_tool": static_filings_tool(PASSAGES), "db": db}
    runs = run_items([item], settings, MemoryStore(tmp_path / "m.sqlite"), "main", **kwargs)
    row = runs.iloc[0]
    assert (
        row.numeric_recall == 1.0 and row.unsupported_numbers == 0 and row.required_specialists_used
    )
    out = tmp_path / "evals"
    out.mkdir()
    summarize(runs).to_csv(out / "quality.csv", index=False)
    fault_injection(runs, settings, use_llm=False).to_csv(out / "faults.csv", index=False)
    readme = tmp_path / "README.md"
    readme.write_text("a\n<!-- RESULTS:START -->\nold\n<!-- RESULTS:END -->\n")
    body = report(out, readme)
    assert "Critic fault injection" in body and "old" not in readme.read_text()


def test_fault_injection_runs_llm_critic_and_eval_stops_on_systematic_failure(
    settings, db, tmp_path
):
    import pytest

    item = EvalItem(
        qid="growth-NVDA-revenue-2025",
        template="growth",
        tickers=["NVDA"],
        requires=["quant"],
        question="How did NVIDIA's revenue change in fiscal 2025 and why?",
        truth=[TruthNumber(label="rev", value=130_497_000_000, unit="USD")],
    )
    kwargs = {"filings_tool": static_filings_tool(PASSAGES), "db": db}
    runs = run_items(
        [item],
        settings,
        MemoryStore(tmp_path / "m.sqlite"),
        "main",
        backend=ScriptedBackend(),
        **kwargs,
    )
    # The critic rejects nothing here, so only code checks catch the corruptions; the point
    # is that the LLM-review path builds valid requests end to end.
    faults = fault_injection(runs, settings, use_llm=True, backend=ScriptedBackend(), **kwargs)
    assert len(faults) and "with_llm" in faults and (faults["llm_cost"] >= 0).all()

    class Broken:
        def create(self, **req):
            raise RuntimeError("400 invalid request")

    with pytest.raises(RuntimeError, match="Stopping the eval"):
        run_items(
            [item, item, item],
            settings,
            MemoryStore(tmp_path / "m2.sqlite"),
            "x",
            backend=Broken(),
            **kwargs,
        )


def test_xbrl_queries_are_thread_safe(db):
    from concurrent.futures import ThreadPoolExecutor

    sql = "SELECT ticker, SUM(value) AS v FROM financials GROUP BY ticker ORDER BY ticker"
    with ThreadPoolExecutor(max_workers=16) as pool:
        outs = list(pool.map(lambda _: db.query(sql), range(200)))
    assert len(set(outs)) == 1


def test_eval_cli_rerun_of_one_question_type_merges_results(settings, db, monkeypatch, tmp_path):
    """`era eval --templates X` re-runs only X and keeps every other saved row."""
    import pandas as pd
    from typer.testing import CliRunner

    import equity_agents.cli as cli
    import equity_agents.evals.run as run_mod
    from equity_agents.evals.dataset import save_eval_set

    out = settings.resolve(settings.paths.eval_dir)
    out.mkdir(parents=True)
    q = "How did NVIDIA's revenue change in fiscal 2025 and why?"
    truth = [TruthNumber(label="rev", value=130_497_000_000, unit="USD")]
    items = [
        EvalItem(
            qid="g1",
            template="growth",
            tickers=["NVDA"],
            requires=["quant"],
            question=q,
            truth=truth,
        ),
        EvalItem(
            qid="d1",
            template="drivers",
            tickers=["NVDA"],
            requires=["quant"],
            question=q,
            truth=truth,
        ),
    ]
    save_eval_set(items, out / "questions.jsonl")
    pd.DataFrame(
        [
            {
                "qid": "g1",
                "template": "growth",
                "run_id": "old-g1",
                "verdict": "approve",
                "numeric_recall": 0.5,
            },
            {
                "qid": "d1",
                "template": "drivers",
                "run_id": "old-d1",
                "verdict": "revise",
                "numeric_recall": 0.0,
            },
        ]
    ).to_csv(out / "runs.csv", index=False)
    pd.DataFrame(
        [
            {
                "run_id": "old-g1",
                "mutation": "drop_citation",
                "deterministic": True,
                "with_llm": True,
                "llm_cost": 0.01,
            },
        ]
    ).to_csv(out / "faults.csv", index=False)

    real_make_deps = run_mod.make_deps

    def fake_make_deps(s, run_id, **kw):
        kw.setdefault("backend", ScriptedBackend())
        return real_make_deps(s, run_id, filings_tool=static_filings_tool(PASSAGES), db=db, **kw)

    monkeypatch.setattr(run_mod, "make_deps", fake_make_deps)
    monkeypatch.setattr(cli, "load_settings", lambda: settings)
    monkeypatch.setattr(run_mod, "ROOT", tmp_path)  # don't touch the real README
    result = CliRunner().invoke(cli.app, ["eval", "--templates", "drivers"])
    assert result.exit_code == 0, result.output
    runs = pd.read_csv(out / "runs.csv").set_index("qid")
    assert runs.loc["g1", "run_id"] == "old-g1" and runs.loc["g1", "numeric_recall"] == 0.5
    assert runs.loc["d1", "run_id"] != "old-d1" and runs.loc["d1", "numeric_recall"] == 1.0
    faults = pd.read_csv(out / "faults.csv")
    assert "old-g1" in set(faults["run_id"]) and len(faults) > 1
    assert set(pd.read_csv(out / "quality.csv")["template"]) >= {"growth", "drivers"}
