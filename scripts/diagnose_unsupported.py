"""Free, offline: for each saved eval memo, list numbers the code check rejected and whether
each one appears in a filing finding's quote (the case the quote check now accepts)."""

import glob
import json
from pathlib import Path

from equity_agents.graph import _strip_sources
from equity_agents.numbers import _in_quotes, unsupported_numbers
from equity_agents.state import Finding

total = in_quote = 0
for d in sorted(glob.glob("runs/eval-main-*")):
    d = Path(d)
    if not (d / "memo.md").exists():
        continue
    findings = [Finding.model_validate(f) for f in json.loads((d / "findings.json").read_text())]
    for f in findings:
        f.source.quote_verified = True  # what the quote check would decide, approximately
    verified = [f for f in findings if f.confidence != "low"]
    bad = unsupported_numbers(_strip_sources((d / "memo.md").read_text()), verified)
    if not bad:
        continue
    hits = [n for n in bad if _in_quotes(n, verified)]
    total += len(bad)
    in_quote += len(hits)
    others = [n.raw for n in bad if n not in hits]
    print(f"{d.name}: {len(bad)} rejected, {len(hits)} found in quotes; other: {others[:8]}")
print(f"\nTOTAL rejected {total}, found in quotes {in_quote}")
