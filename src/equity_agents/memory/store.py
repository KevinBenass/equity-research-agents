"""Long-term memory, in three kinds, persisted in one SQLite file.

semantic    Durable facts with provenance: "NVDA revenue FY2025 = 130.5B (xbrl:...)". Written
            from verified findings after a memo is approved, read by the planner and by the
            quant analyst (a `recall_facts` tool), so known numbers aren't re-fetched.
            Conflicting values for the same key are versioned, never silently overwritten.
episodic    Past research runs: question, summary, tickers, cost. Retrieved by similarity so
            the planner can build on prior work instead of starting cold.
procedural  Lessons distilled from critic reviews ("cite a source for every growth rate").
            Injected into the writer's prompt, so recurring mistakes get rarer over time.

Working memory is the LangGraph state of the current run; it is not stored here.
"""

from __future__ import annotations

import json
import sqlite3
import threading
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from equity_agents.state import Finding, Issue

SCHEMA = """
CREATE TABLE IF NOT EXISTS facts (
    key TEXT, version INTEGER, ticker TEXT, metric TEXT, period TEXT, value REAL, unit TEXT,
    claim TEXT, ref TEXT, source_kind TEXT, run_id TEXT, created TEXT, superseded INTEGER DEFAULT 0,
    times_confirmed INTEGER DEFAULT 1, PRIMARY KEY (key, version)
);
CREATE TABLE IF NOT EXISTS episodes (
    run_id TEXT PRIMARY KEY, question TEXT, summary TEXT, tickers TEXT, created TEXT,
    cost_usd REAL, n_findings INTEGER
);
CREATE TABLE IF NOT EXISTS lessons (
    lesson TEXT PRIMARY KEY, count INTEGER, last_run TEXT
);
"""

LESSON_TEXT = {
    "unsupported_number": "Every number in the memo must match a finding exactly (same value, unit and period); compute derived figures with the calculator and record them as findings.",
    "missing_citation": "Cite a finding id like [F3] in every sentence that states a fact or a number.",
    "unknown_citation": "Only cite finding ids that exist in the findings list.",
    "unsupported_claim": "Do not generalize beyond what the findings say; qualify claims that rest on one source.",
    "missing_evidence": "Plan both a quantitative and a qualitative task when a question asks 'why' or 'how'.",
}


class MemoryStore:
    def __init__(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        self._db = sqlite3.connect(path, check_same_thread=False)
        self._db.executescript(SCHEMA)
        self._lock = threading.RLock()

    # ------------------------------------------------------------------ semantic
    def _query(self, sql: str, args: tuple | list = ()) -> list[tuple]:
        """Read under the lock: parallel workers share this SQLite connection."""
        with self._lock:
            return self._db.execute(sql, args).fetchall()

    @staticmethod
    def fact_key(f: Finding) -> str:
        if f.metric and f.ticker and f.period:
            return f"{f.ticker}|{f.metric}|{f.period}"
        return f"{f.ticker or '-'}|claim|{f.claim.lower()[:120]}"

    def remember_facts(self, findings: list[Finding], run_id: str) -> dict[str, int]:
        """Store verified findings. Same value -> confirm; different value -> new version."""
        stats = Counter()
        now = datetime.now(UTC).isoformat()
        with self._lock:
            for f in findings:
                if f.source.kind not in ("xbrl", "filing") or f.confidence == "low":
                    continue  # derived numbers are recomputed, not remembered
                key = self.fact_key(f)
                row = self._db.execute(
                    "SELECT version, value FROM facts WHERE key=? AND superseded=0", (key,)
                ).fetchone()
                if row and (
                    row[1] == f.value
                    or (
                        row[1] is not None
                        and f.value is not None
                        and abs(row[1] - f.value) <= 1e-9 * max(1, abs(f.value))
                    )
                ):
                    self._db.execute(
                        "UPDATE facts SET times_confirmed = times_confirmed + 1 WHERE key=? AND version=?",
                        (key, row[0]),
                    )
                    stats["confirmed"] += 1
                    continue
                version = 1
                if row:
                    self._db.execute(
                        "UPDATE facts SET superseded=1 WHERE key=? AND version=?", (key, row[0])
                    )
                    version = row[0] + 1
                    stats["conflicts"] += 1
                self._db.execute(
                    "INSERT INTO facts (key, version, ticker, metric, period, value, unit, claim, ref, source_kind, run_id, created) "
                    "VALUES (?,?,?,?,?,?,?,?,?,?,?,?)",
                    (
                        key,
                        version,
                        f.ticker,
                        f.metric,
                        f.period,
                        f.value,
                        f.unit,
                        f.claim,
                        f.source.ref,
                        f.source.kind,
                        run_id,
                        now,
                    ),
                )
                stats["new"] += 1
            self._db.commit()
        return dict(stats)

    def recall_facts(
        self, tickers: list[str] | None = None, metric: str | None = None, limit: int = 20
    ) -> list[dict]:
        sql = "SELECT ticker, metric, period, value, unit, claim, ref, times_confirmed FROM facts WHERE superseded=0"
        args: list = []
        if tickers:
            sql += f" AND ticker IN ({','.join('?' * len(tickers))})"
            args += [t.upper() for t in tickers]
        if metric:
            sql += " AND metric = ?"
            args.append(metric)
        sql += " ORDER BY times_confirmed DESC, created DESC LIMIT ?"
        args.append(limit)
        cols = ["ticker", "metric", "period", "value", "unit", "claim", "ref", "times_confirmed"]
        return [dict(zip(cols, r, strict=True)) for r in self._query(sql, args)]

    # ------------------------------------------------------------------ episodic
    def remember_episode(
        self,
        run_id: str,
        question: str,
        summary: str,
        tickers: list[str],
        cost: float,
        n_findings: int,
    ) -> None:
        with self._lock:
            self._db.execute(
                "INSERT OR REPLACE INTO episodes VALUES (?,?,?,?,?,?,?)",
                (
                    run_id,
                    question,
                    summary,
                    json.dumps(tickers),
                    datetime.now(UTC).isoformat(),
                    cost,
                    n_findings,
                ),
            )
            self._db.commit()

    def similar_episodes(self, question: str, k: int = 2, min_score: float = 0.15) -> list[dict]:
        rows = self._query("SELECT run_id, question, summary, created FROM episodes")
        if not rows:
            return []
        corpus = [f"{r[1]} {r[2]}" for r in rows]
        vec = TfidfVectorizer(stop_words="english", ngram_range=(1, 2)).fit(corpus + [question])
        sims = cosine_similarity(vec.transform([question]), vec.transform(corpus))[0]
        ranked = sorted(zip(sims, rows, strict=True), key=lambda x: -x[0])[:k]
        return [
            {
                "run_id": r[0],
                "question": r[1],
                "summary": r[2],
                "created": r[3],
                "similarity": round(float(s), 3),
            }
            for s, r in ranked
            if s >= min_score
        ]

    # ------------------------------------------------------------------ procedural
    def learn_from_review(self, issues: list[Issue], run_id: str) -> None:
        with self._lock:
            for kind in {i.kind for i in issues}:
                lesson = LESSON_TEXT.get(kind)
                if not lesson:
                    continue
                self._db.execute(
                    "INSERT INTO lessons VALUES (?, 1, ?) ON CONFLICT(lesson) DO UPDATE SET count = count + 1, last_run = excluded.last_run",
                    (lesson, run_id),
                )
            self._db.commit()

    def lessons(self, k: int = 5) -> list[str]:
        return [
            r[0]
            for r in self._query("SELECT lesson FROM lessons ORDER BY count DESC LIMIT ?", (k,))
        ]

    def stats(self) -> dict[str, int]:
        q = lambda sql: self._query(sql)[0][0]  # noqa: E731
        return {
            "facts": q("SELECT count(*) FROM facts WHERE superseded=0"),
            "superseded_facts": q("SELECT count(*) FROM facts WHERE superseded=1"),
            "episodes": q("SELECT count(*) FROM episodes"),
            "lessons": q("SELECT count(*) FROM lessons"),
        }
