"""Evaluation runner: answer quality, critic fault injection, and a memory ablation."""

from __future__ import annotations

import json
import random
import re
import shutil
from pathlib import Path

import pandas as pd
from rich.console import Console

from equity_agents.config import ROOT, Settings
from equity_agents.evals.dataset import EvalItem
from equity_agents.evals.fault_injection import MUTATIONS, deterministic_detects, mutate
from equity_agents.evals.metrics import score_run
from equity_agents.graph import review_memo
from equity_agents.memory.store import MemoryStore
from equity_agents.runner import make_deps, research
from equity_agents.state import Finding
from equity_agents.tracing import load_trace

console = Console()
START, END = "<!-- RESULTS:START -->", "<!-- RESULTS:END -->"


def run_items(
    items: list[EvalItem], settings: Settings, memory: MemoryStore, label: str, **dep_kwargs
) -> pd.DataFrame:
    rows = []
    for i, item in enumerate(items, 1):
        console.rule(f"[{label}] {i}/{len(items)} {item.qid}")
        run_id = f"eval-{label}-{item.qid}"[:80]
        shutil.rmtree(settings.resolve(settings.paths.runs_dir) / run_id, ignore_errors=True)
        deps = make_deps(settings, run_id, memory=memory, **dep_kwargs)
        try:
            out = research(item.question, settings, run_id=run_id, deps=deps)
        except Exception as exc:
            console.print(f"[red]run failed: {exc}")
            rows.append(
                {"qid": item.qid, "template": item.template, "error": str(exc), "pass": label}
            )
            continue
        findings = [
            Finding.model_validate(f)
            for f in json.loads((out.run_dir / "findings.json").read_text())
        ]
        summary = json.loads((out.run_dir / "summary.json").read_text())
        trace = (
            load_trace(out.run_dir / "trace.jsonl")
            if (out.run_dir / "trace.jsonl").exists()
            else deps.llm.tracer.events
        )
        row = score_run(item, out.memo, findings, summary, trace)
        row.update(run_id=out.run_id, run_dir=str(out.run_dir), **{"pass": label})
        rows.append(row)
        console.print(
            {
                k: row[k]
                for k in (
                    "numeric_recall",
                    "unsupported_numbers",
                    "verdict",
                    "cost_usd",
                    "latency_s",
                )
            }
        )
    return pd.DataFrame(rows)


def fault_injection(
    runs: pd.DataFrame, settings: Settings, use_llm: bool, seed: int = 0, **dep_kwargs
) -> pd.DataFrame:
    rng = random.Random(seed)
    rows = []
    for r in runs[runs["verdict"] == "approve"].itertuples():
        run_dir = Path(r.run_dir)
        memo = (run_dir / "memo.md").read_text()
        findings = [
            Finding.model_validate(f) for f in json.loads((run_dir / "findings.json").read_text())
        ]
        question = json.loads((run_dir / "summary.json").read_text())["question"]
        companies = sorted(
            {
                w
                for w in re.findall(r"\b[A-Z][A-Za-z&]+(?:\s[A-Z][A-Za-z&]+)?\b", memo)
                if w
                in {
                    "Apple",
                    "Microsoft",
                    "NVIDIA",
                    "Nvidia",
                    "Amazon",
                    "Alphabet",
                    "JPMorgan",
                    "Exxon",
                }
            }
            | {"Apple", "Microsoft", "Amazon"}
        )
        deps = (
            make_deps(settings, f"fault-{r.run_id}", trace=False, **dep_kwargs) if use_llm else None
        )
        for kind in MUTATIONS:
            corrupted = mutate(memo, kind, findings, rng, companies)
            if corrupted is None or corrupted == memo:
                continue
            row = {
                "run_id": r.run_id,
                "mutation": kind,
                "deterministic": deterministic_detects(corrupted, findings),
            }
            if use_llm and deps is not None:
                review = review_memo(deps, question, corrupted, findings)
                row["with_llm"] = review.verdict != "approve"
                row["llm_cost"] = deps.llm.tracer.totals()["cost_usd"]
            rows.append(row)
    return pd.DataFrame(rows)


def summarize(df: pd.DataFrame) -> pd.DataFrame:
    ok = df[df.get("error").isna()] if "error" in df else df
    agg = (
        ok.groupby("template")
        .agg(
            n=("qid", "count"),
            numeric_recall=("numeric_recall", "mean"),
            all_correct=("all_numbers_correct", "mean"),
            unsupported_numbers=("unsupported_numbers", "mean"),
            citation_coverage=("citation_coverage", "mean"),
            specialists_ok=("required_specialists_used", "mean"),
            approved=("verdict", lambda v: (v == "approve").mean()),
            revisions=("revisions", "mean"),
            tool_calls=("tool_calls", "mean"),
            cost_usd=("cost_usd", "mean"),
            latency_s=("latency_s", "mean"),
        )
        .reset_index()
    )
    total = ok.agg(
        {
            "numeric_recall": "mean",
            "all_numbers_correct": "mean",
            "unsupported_numbers": "mean",
            "citation_coverage": "mean",
            "required_specialists_used": "mean",
            "revisions": "mean",
            "tool_calls": "mean",
            "cost_usd": "mean",
            "latency_s": "mean",
        }
    )
    total_row = {
        "template": "**all**",
        "n": len(ok),
        "numeric_recall": total["numeric_recall"],
        "all_correct": total["all_numbers_correct"],
        "unsupported_numbers": total["unsupported_numbers"],
        "citation_coverage": total["citation_coverage"],
        "specialists_ok": total["required_specialists_used"],
        "approved": (ok["verdict"] == "approve").mean(),
        "revisions": total["revisions"],
        "tool_calls": total["tool_calls"],
        "cost_usd": total["cost_usd"],
        "latency_s": total["latency_s"],
    }
    return pd.concat([agg, pd.DataFrame([total_row])], ignore_index=True)


def report(out_dir: Path, readme: Path | None = None) -> str:
    def fmt(df: pd.DataFrame) -> str:
        d = df.copy()
        for c in d.columns:
            if d[c].dtype.kind == "f":
                d[c] = d[c].map(
                    lambda v, c=c: (
                        "" if pd.isna(v) else (f"${v:.3f}" if c.startswith("cost") else f"{v:.2f}")
                    )
                )
        return d.to_markdown(index=False, disable_numparse=True)

    parts = []
    if (out_dir / "quality.csv").exists():
        parts += [
            "#### Answer quality and cost by question type",
            "",
            fmt(pd.read_csv(out_dir / "quality.csv")),
            "",
        ]
    if (out_dir / "faults.csv").exists():
        f = pd.read_csv(out_dir / "faults.csv")
        cols = {"deterministic": "caught by code checks"}
        if "with_llm" in f:
            cols["with_llm"] = "caught by code + LLM review"
        t = f.groupby("mutation")[list(cols)].mean().rename(columns=cols).reset_index()
        t.insert(1, "n", f.groupby("mutation").size().values)
        parts += ["#### Critic fault injection: share of corrupted memos rejected", "", fmt(t), ""]
    if (out_dir / "memory.csv").exists():
        m = pd.read_csv(out_dir / "memory.csv")
        parts += ["#### Memory: same questions, cold vs. warm memory", "", fmt(m), ""]
    body = "\n".join(parts) or "_No results yet._"
    (out_dir / "REPORT.md").write_text("# Evaluation results\n\n" + body + "\n")
    readme = readme or ROOT / "README.md"
    if readme.exists() and START in readme.read_text():
        readme.write_text(
            re.sub(
                re.escape(START) + ".*?" + re.escape(END),
                lambda _: f"{START}\n{body}\n{END}",
                readme.read_text(),
                flags=re.DOTALL,
            )
        )
    return body


def memory_ablation(
    items: list[EvalItem], settings: Settings, out_dir: Path, **dep_kwargs
) -> pd.DataFrame:
    db_path = out_dir / "memory_ablation.sqlite"
    if db_path.exists():
        db_path.unlink()
    memory = MemoryStore(db_path)
    cold = run_items(items, settings, memory, "cold", **dep_kwargs)
    warm = run_items(items, settings, memory, "warm", **dep_kwargs)
    rows = []
    for label, df in (("cold (empty memory)", cold), ("warm (after one pass)", warm)):
        rows.append(
            {
                "memory": label,
                "tool_calls": df["tool_calls"].mean(),
                "llm_calls": df["llm_calls"].mean(),
                "cost_usd": df["cost_usd"].mean(),
                "latency_s": df["latency_s"].mean(),
                "numeric_recall": df["numeric_recall"].mean(),
            }
        )
    return pd.DataFrame(rows)
