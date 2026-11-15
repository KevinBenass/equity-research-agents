"""Deterministic number checking: every figure in a memo must be backed by a finding.

The critic's first pass is code, not a model: numbers are parsed from the memo (with their
displayed precision, so "$130.5 billion" matches 130,497,000,000 but not 131.2B) and matched
against finding values. LLM review then only has to judge claims that code cannot.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from equity_agents.state import Finding

SCALE = {
    "trillion": 1e12,
    "tn": 1e12,
    "t": 1e12,
    "billion": 1e9,
    "bn": 1e9,
    "b": 1e9,
    "million": 1e6,
    "mn": 1e6,
    "mm": 1e6,
    "m": 1e6,
    "thousand": 1e3,
    "k": 1e3,
}

NUM_RE = re.compile(
    r"(?P<neg>[-−(])?(?P<cur>\$|USD\s?)?(?P<num>\d{1,3}(?:,\d{3})+(?:\.\d+)?|\d+(?:\.\d+)?)"
    r"(?:\s?(?P<pct>%|percent|pp|percentage points)|\s?(?P<scale>trillion|billion|million|thousand|tn|bn|mn|mm|[tbmk])\b)?",
    re.IGNORECASE,
)
CITE_RE = re.compile(r"\[(F\d+)\]")


@dataclass
class MemoNumber:
    raw: str
    value: float
    kind: str  # "pct" | "money" | "plain"
    tolerance: float
    sentence: str


def _decimals(num: str) -> int:
    return len(num.split(".")[1]) if "." in num else 0


def extract_numbers(text: str) -> list[MemoNumber]:
    text_wo_cites = CITE_RE.sub("", text)
    out: list[MemoNumber] = []
    for sentence in re.split(r"(?<=[.!?])\s+|\n", text_wo_cites):
        for m in NUM_RE.finditer(sentence):
            num, pct, scale, cur = m.group("num"), m.group("pct"), m.group("scale"), m.group("cur")
            before = sentence[max(0, m.start() - 3) : m.start()]
            after = sentence[m.end() : m.end() + 2]
            # Skip identifiers such as "10-K", "Item 7A", "Q4", "S-1": digits glued to letters.
            if re.match(r"-?[A-Za-z]", after) or (before[-1:].isalnum() and not cur):
                continue
            plain = num.replace(",", "")
            # Skip years, fiscal-year labels, list markers and footnote-like small integers.
            if not pct and not scale and not cur:
                if re.fullmatch(r"(19|20)\d{2}", plain) or before.upper().endswith(("FY", "Q")):
                    continue
                if "." not in plain and float(plain) < 10:
                    continue
            mult = SCALE.get((scale or "").lower(), 1.0)
            value = float(plain) * mult
            if m.group("neg") in ("-", "−", "("):
                value = -value
            tol = 0.5 * 10 ** (-_decimals(num)) * mult
            kind = "pct" if pct else ("money" if cur or scale else "plain")
            out.append(MemoNumber(m.group(0).strip(), value, kind, tol, sentence.strip()))
    return out


def _in_quotes(n: MemoNumber, findings: list[Finding]) -> bool:
    """Figures stated in a quote that code confirmed verbatim in a filing passage."""
    for f in findings:
        if not (f.source.quote_verified and f.confidence != "low"):
            continue
        for q in extract_numbers(f.source.quote):
            if q.kind != n.kind and "plain" not in (q.kind, n.kind):
                continue
            # Match at the coarser of the two precisions ("$115,186 million" ~ "$115.2 billion").
            tol = max(n.tolerance, q.tolerance) + 1e-9 * abs(q.value)
            if abs(abs(n.value) - abs(q.value)) <= tol:
                return True
    return False


def supported(n: MemoNumber, findings: list[Finding]) -> bool:
    if _in_quotes(n, findings):
        return True
    for f in findings:
        if f.value is None:
            continue
        candidates = [f.value]
        if f.unit == "ratio":
            candidates.append(f.value * 100)  # 0.624 shown as 62.4%
        if f.unit == "%":
            candidates.append(f.value)
        for c in candidates:
            for v in (n.value, abs(n.value)):
                if abs(v - c) <= n.tolerance + 1e-9 * abs(c):
                    return True
                if abs(abs(v) - abs(c)) <= n.tolerance + 1e-9 * abs(c):
                    return True  # "declined 5.2%" vs a finding of -5.2
    return False


def unsupported_numbers(memo: str, findings: list[Finding]) -> list[MemoNumber]:
    return [n for n in extract_numbers(memo) if not supported(n, findings)]


def citation_problems(memo: str, findings: list[Finding]) -> tuple[list[str], list[str]]:
    """(sentences with numbers but no citation, cited ids that don't exist)."""
    ids = {f.id for f in findings}
    uncited, unknown = [], set()
    body = memo.split("## Sources")[0]
    for sentence in re.split(r"(?<=[.!?])\s+|\n", body):
        s = sentence.strip()
        if not s or s.startswith("#") or s.startswith("|---"):
            continue
        cites = CITE_RE.findall(s)
        unknown |= {c for c in cites if c not in ids}
        if not cites and extract_numbers(s):
            uncited.append(s[:200])
    return uncited, sorted(unknown)
