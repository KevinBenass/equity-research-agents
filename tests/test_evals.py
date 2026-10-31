from fakes import PASSAGES, ScriptedBackend

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
        def create(self, **req):
            self.requests.append(req)
            return {
                "content": [
                    {
                        "type": "tool_use",
                        "id": "t1",
                        "name": "submit_findings",
                        "input": {
                            "findings": [
                                {
                                    "claim": "Revenue",
                                    "ticker": "NVDA",
                                    "metric": "revenue",
                                    "value": 1.0,
                                    "unit": "USD",
                                    "period": "FY2025",
                                    "source": {"kind": "xbrl", "ref": "xbrl:NVDA:revenue:FY2025"},
                                }
                            ]
                        },
                    }
                ],
                "stop_reason": "tool_use",
                "usage": {"input_tokens": 10, "output_tokens": 10},
            }

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
    f = report_.findings[0]
    assert (
        f.confidence == "low"
        and "never returned by a tool" in f.claim
        and "does not match XBRL" in f.claim
    )


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
