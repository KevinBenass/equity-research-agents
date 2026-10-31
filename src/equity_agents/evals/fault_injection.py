"""Does the critic catch errors? Corrupt approved memos on purpose and see.

A reviewer that approves everything looks identical to a good one on clean drafts. So each
approved memo is mutated in known ways and passed back through the critic's checks:

  perturb_number    one figure changed by +7% (a plausible-looking wrong number)
  fabricate_number  an extra sentence with an invented figure, citing a real finding
  drop_citation     citations removed from a sentence that states a number
  unknown_citation  a citation to a finding that does not exist
  swap_company      a company name replaced by another one (needs judgment: LLM check only)

Deterministic checks cover the first four by construction; the report shows whether the
LLM review adds coverage for the fifth, and what it costs.
"""

from __future__ import annotations

import random
import re

from equity_agents.graph import _strip_sources
from equity_agents.numbers import NUM_RE, citation_problems, extract_numbers, unsupported_numbers
from equity_agents.state import Finding

MUTATIONS = [
    "perturb_number",
    "fabricate_number",
    "drop_citation",
    "unknown_citation",
    "swap_company",
]


def mutate(
    memo: str, kind: str, findings: list[Finding], rng: random.Random, companies: list[str]
) -> str | None:
    body = _strip_sources(memo)
    tail = memo[len(body) :]
    if kind == "perturb_number":
        matches = [
            m for m in NUM_RE.finditer(body) if m.group("pct") or m.group("scale") or m.group("cur")
        ]
        if not matches:
            return None
        m = rng.choice(matches)
        num = m.group("num")
        decimals = len(num.split(".")[1]) if "." in num else 0
        new = f"{float(num.replace(',', '')) * 1.07:,.{decimals}f}"
        return body[: m.start("num")] + new + body[m.end("num") :] + tail
    if kind == "fabricate_number":
        cite = findings[0].id if findings else "F1"
        return (
            body.replace(
                "## Analysis\n",
                f"## Analysis\nThe company also returned $47.3 billion to shareholders [{cite}].\n",
                1,
            )
            + tail
        )
    if kind == "drop_citation":
        lines = body.split("\n")
        idx = [
            i for i, ln in enumerate(lines) if re.search(r"\[F\d+\]", ln) and extract_numbers(ln)
        ]
        if not idx:
            return None
        i = rng.choice(idx)
        lines[i] = re.sub(r"\s?\[F\d+\]", "", lines[i])
        return "\n".join(lines) + tail
    if kind == "unknown_citation":
        return re.sub(r"\[F(\d+)\]", "[F99]", body, count=1) + tail
    if kind == "swap_company":
        present = [c for c in companies if c in body]
        others = [c for c in companies if c not in body]
        if not present or not others:
            return None
        return body.replace(present[0], rng.choice(others), 1) + tail
    raise ValueError(kind)


def deterministic_detects(memo: str, findings: list[Finding]) -> bool:
    uncited, unknown = citation_problems(memo, findings)
    return bool(unsupported_numbers(_strip_sources(memo), findings) or uncited or unknown)
