"""Command line entry point: `era --help`."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Annotated

import typer
from rich.console import Console
from rich.markdown import Markdown
from rich.table import Table
from rich.tree import Tree

from equity_agents.config import load_settings

app = typer.Typer(add_completion=False, help="Equity research agent team (LangGraph + Claude).")
console = Console()


@app.command()
def data() -> None:
    """Download SEC XBRL company facts for the configured universe."""
    from equity_agents.tools.xbrl import FinancialsDB, download_facts

    s = load_settings()
    cache = s.resolve(s.tools.xbrl_cache_dir)
    download_facts(s.universe.tickers, cache, s.sec_user_agent, s.universe.ciks)
    db = FinancialsDB(cache, s.universe.tickers)
    console.print(db.df.groupby("ticker")["metric"].nunique().rename("metrics").to_frame())


@app.command()
def ask(
    question: str,
    review: Annotated[
        bool, typer.Option(help="Pause for human approval before publishing")
    ] = False,
) -> None:
    """Research a question end to end and print the memo."""
    from equity_agents.runner import research

    s = load_settings(overrides={"orchestration.human_review": review})
    out = research(question, s)
    if out.interrupted:
        console.print(Markdown(out.state["__interrupt__"][0].value["memo"]))
        console.print(
            f"\n[yellow]Paused for review. Approve with: era resume {out.run_id} --approve "
            f"(or --feedback 'what to change')"
        )
        return
    console.print(Markdown(out.memo))
    t = out.totals
    console.print(
        f"[dim]run {out.run_id} · {t['llm_calls']} LLM calls · {t['tool_calls']} tool calls · "
        f"${t['cost_usd']:.3f} · {out.latency_s:.0f}s · {out.run_dir}"
    )


@app.command()
def resume(run_id: str, approve: bool = False, feedback: str = "") -> None:
    """Resume a run paused for human review."""
    from equity_agents.runner import resume as _resume

    s = load_settings(overrides={"orchestration.human_review": True})
    out = _resume(run_id, s, approve=approve, feedback=feedback)
    console.print(Markdown(out.memo))


@app.command()
def trace(run_id: str) -> None:
    """Show a run's trace as a tree with per-agent cost."""
    from equity_agents.tracing import load_trace

    s = load_settings()
    run_dir = s.resolve(s.paths.runs_dir) / run_id
    events = load_trace(run_dir / "trace.jsonl")
    summary = json.loads((run_dir / "summary.json").read_text())
    root = Tree(f"[bold]{summary['question']}")
    for e in events:
        if e["kind"] == "llm":
            root.add(
                f"{e['agent']} · {e['model']} · in {e['input_tokens']} / out {e['output_tokens']} tok · ${e['cost_usd']:.4f} · {e['ms']:.0f} ms"
            )
        else:
            flag = " [red]error" if e.get("is_error") else ""
            root.add(
                f"  ↳ {e['name']}({json.dumps(e.get('args', {}))[:80]}) · {e['ms']:.0f} ms{flag}"
            )
    console.print(root)
    table = Table("agent", "calls", "input tok", "output tok", "cost")
    for agent, a in summary["by_agent"].items():
        table.add_row(
            agent,
            f"{a['calls']:.0f}",
            f"{a['input_tokens']:.0f}",
            f"{a['output_tokens']:.0f}",
            f"${a['cost_usd']:.4f}",
        )
    console.print(table)


@app.command()
def memory() -> None:
    """Show what the team remembers."""
    from equity_agents.memory.store import MemoryStore

    s = load_settings()
    m = MemoryStore(s.resolve(s.paths.memory_db))
    console.print(m.stats())
    for lesson in m.lessons():
        console.print(f"• {lesson}")


@app.command("eval")
def evaluate(
    limit: Annotated[int | None, typer.Option(help="Use only the first N questions")] = None,
    llm_critic_faults: Annotated[
        bool, typer.Option(help="Also run the LLM critic on corrupted memos")
    ] = True,
    memory_ablation: Annotated[
        bool, typer.Option(help="Run the cold vs warm memory comparison")
    ] = True,
    templates: Annotated[
        str | None,
        typer.Option(
            help="Re-run only these question types (comma-separated) and merge them into the "
            "saved results; other rows, their fault-injection rows and the memory table are kept"
        ),
    ] = None,
) -> None:
    """Build the eval set from XBRL, run the team, score, fault-inject, write the report."""
    from equity_agents.evals.dataset import build_eval_set, load_eval_set, save_eval_set
    from equity_agents.evals.run import fault_injection, report, run_items, summarize
    from equity_agents.evals.run import memory_ablation as _mem_ablation
    from equity_agents.memory.store import MemoryStore
    from equity_agents.tools.xbrl import FinancialsDB

    s = load_settings()
    out_dir = s.resolve(s.paths.eval_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    items_path = out_dir / "questions.jsonl"
    if not items_path.exists():
        db = FinancialsDB(s.resolve(s.tools.xbrl_cache_dir), s.universe.tickers)
        save_eval_set(build_eval_set(db, s.universe.fiscal_years), items_path)
    items = load_eval_set(items_path)[:limit]

    if templates:
        import pandas as pd

        wanted = {t.strip() for t in templates.split(",")}
        unknown = wanted - {i.template for i in items}
        if unknown:
            raise typer.BadParameter(f"Unknown question types: {sorted(unknown)}")
        items = [i for i in items if i.template in wanted]
        old_runs = pd.read_csv(out_dir / "runs.csv")
        old_faults = pd.read_csv(out_dir / "faults.csv")
        new = run_items(items, s, MemoryStore(out_dir / "eval_memory.sqlite"), "main")
        runs = pd.concat([old_runs[~old_runs["template"].isin(wanted)], new], ignore_index=True)
        kept = old_faults[
            old_faults["run_id"].isin(runs.loc[~runs["template"].isin(wanted), "run_id"])
        ]
        faults = pd.concat(
            [kept, fault_injection(new, s, use_llm=llm_critic_faults)], ignore_index=True
        )
        runs.to_csv(out_dir / "runs.csv", index=False)
        summarize(runs).to_csv(out_dir / "quality.csv", index=False)
        faults.to_csv(out_dir / "faults.csv", index=False)
        console.print(Markdown(report(out_dir)))
        return

    runs = run_items(items, s, MemoryStore(out_dir / "eval_memory.sqlite"), "main")
    runs.to_csv(out_dir / "runs.csv", index=False)
    summarize(runs).to_csv(out_dir / "quality.csv", index=False)
    fault_injection(runs, s, use_llm=llm_critic_faults).to_csv(out_dir / "faults.csv", index=False)
    if memory_ablation:
        subset = [i for i in items if i.template in ("growth", "margin")][:6]
        _mem_ablation(subset, s, out_dir).to_csv(out_dir / "memory.csv", index=False)
    console.print(Markdown(report(out_dir)))


@app.command()
def smoke() -> None:
    """Cheap end-to-end check before the full eval: one question per type (about $0.50).

    Runs against the real API, stops at the first error, and writes to <eval_dir>/smoke so
    the full results and the README are untouched.
    """
    from equity_agents.evals.dataset import build_eval_set, load_eval_set, save_eval_set
    from equity_agents.evals.run import fault_injection, run_items, summarize
    from equity_agents.memory.store import MemoryStore
    from equity_agents.tools.xbrl import FinancialsDB

    s = load_settings()
    eval_dir = s.resolve(s.paths.eval_dir)
    eval_dir.mkdir(parents=True, exist_ok=True)
    items_path = eval_dir / "questions.jsonl"
    if not items_path.exists():
        db = FinancialsDB(s.resolve(s.tools.xbrl_cache_dir), s.universe.tickers)
        save_eval_set(build_eval_set(db, s.universe.fiscal_years), items_path)
    items, seen = [], set()
    for item in load_eval_set(items_path):
        if item.template not in seen:
            seen.add(item.template)
            items.append(item)
    out_dir = eval_dir / "smoke"
    out_dir.mkdir(exist_ok=True)
    memory_path = out_dir / "memory.sqlite"
    memory_path.unlink(missing_ok=True)
    runs = run_items(items, s, MemoryStore(memory_path), "smoke", fail_fast=True)
    runs.to_csv(out_dir / "runs.csv", index=False)
    console.print(summarize(runs).to_markdown(index=False))
    faults = fault_injection(runs, s, use_llm=True)  # exercises the LLM critic path too
    if len(faults):
        console.print(
            f"Fault injection: {len(faults)} corrupted memos, "
            f"{faults['with_llm'].mean():.0%} rejected by code + LLM review"
        )
    total = runs["cost_usd"].sum() + (faults["llm_cost"].sum() if len(faults) else 0.0)
    console.print(f"[green]Smoke test passed[/] · {len(runs)} questions · ${total:.2f}")


@app.command("eval-report")
def eval_report() -> None:
    """Rebuild the README results block from saved eval CSVs."""
    from equity_agents.evals.run import report

    s = load_settings()
    console.print(Markdown(report(s.resolve(s.paths.eval_dir))))


@app.command()
def graph_diagram(output: Path = Path("docs/graph.mmd")) -> None:
    """Export the LangGraph topology as a Mermaid diagram."""
    from equity_agents.graph import Deps, build_graph
    from equity_agents.llm import LLM
    from equity_agents.memory.store import MemoryStore
    from equity_agents.tools.calc import CALC_TOOL
    from equity_agents.tools.xbrl import FinancialsDB
    from equity_agents.tracing import Tracer

    s = load_settings()
    deps = Deps(
        s,
        LLM(s.models.prices, Tracer("diagram")),
        MemoryStore(Path("/tmp/era-diagram.sqlite")),
        FinancialsDB(Path("/nonexistent"), []),
        CALC_TOOL,
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(build_graph(deps).get_graph().draw_mermaid())
    console.print(f"[green]{output}")


if __name__ == "__main__":
    app()
