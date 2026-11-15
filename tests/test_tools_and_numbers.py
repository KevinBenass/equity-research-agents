import pytest

from equity_agents.numbers import citation_problems, extract_numbers, unsupported_numbers
from equity_agents.state import Finding, Source
from equity_agents.tools.base import ToolError
from equity_agents.tools.calc import safe_eval


def test_xbrl_normalization_rules(db):
    assert db.get("NVDA", "revenue", 2025) == 130_497_000_000
    assert db.get("NVDA", "revenue", 2024) == 60_922_000_000  # latest filing wins over older one
    assert db.get("MSFT", "revenue", 2025) == 281_724_000_000  # higher-priority concept wins
    assert db.get("NVDA", "total_assets", 2025) == 111_601_000_000  # instant concept kept
    nvda_rev = db.df[(db.df.ticker == "NVDA") & (db.df.metric == "revenue")]
    assert sorted(nvda_rev.fiscal_year) == [2023, 2024, 2025]  # no quarterly or 10-Q rows
    assert db.names["NVDA"] == "NVIDIA CORP"


def test_get_metric_returns_citable_refs(db):
    out = db.get_metric("nvda", "revenue", [2025])
    assert "xbrl:NVDA:revenue:FY2025" in out
    with pytest.raises(ToolError):
        db.get_metric("NVDA", "ebitda")


def test_sql_tool_is_read_only(db):
    assert "281724000000" in db.query(
        "SELECT value FROM financials WHERE ticker='MSFT' AND metric='revenue' AND fiscal_year=2025"
    )
    for bad in [
        "DROP TABLE financials",
        "SELECT * FROM read_csv('x')",
        "WITH x AS (SELECT 1) DELETE FROM financials",
        "COPY financials TO 'x'",
    ]:
        with pytest.raises(ToolError):
            db.query(bad)
    with pytest.raises(ToolError, match="SQL error"):
        db.query("SELECT nope FROM financials")


def test_calculator():
    assert safe_eval("(130,497 - 60,922) / 60,922 * 100") == pytest.approx(114.2, abs=0.01)
    assert safe_eval("max(1, 2) + round(2.345, 2)") == pytest.approx(4.35)
    for bad in ["__import__('os')", "open('x')", "2 ** 1000", "1 / 0"]:
        with pytest.raises(ToolError):
            safe_eval(bad)


def _f(fid, value, unit="USD", conf="high"):
    return Finding(
        id=fid,
        claim="x",
        value=value,
        unit=unit,
        source=Source(kind="xbrl", ref="r"),
        confidence=conf,
    )


def test_number_extraction_and_precision_matching():
    nums = extract_numbers(
        "Revenue was $130.5 billion in fiscal 2025, up 114.2% [F1]. Margin: 62.4%. EPS $2.94."
    )
    assert [n.kind for n in nums] == ["money", "pct", "pct", "money"]  # the year is skipped
    findings = [
        _f("F1", 130_497_000_000),
        _f("F2", 114.2049, "%"),
        _f("F3", 0.6242, "ratio"),
        _f("F4", 2.94, "USD/shares"),
    ]
    assert (
        unsupported_numbers(
            "Revenue was $130.5 billion, up 114.2%, margin 62.4%, EPS $2.94.", findings
        )
        == []
    )
    bad = unsupported_numbers("Revenue was $131.2 billion.", findings)
    assert [b.raw for b in bad] == ["$131.2 billion"]
    assert unsupported_numbers("Revenue declined 5.2%.", [_f("F1", -5.2, "%")]) == []


def test_citation_problems():
    findings = [_f("F1", 1.0)]
    memo = "## Summary\nRevenue was $1.0 billion [F1]. Margin was 40.0%.\nSee [F7].\n## Sources\n- [F1] x"
    uncited, unknown = citation_problems(memo, findings)
    assert uncited == ["Margin was 40.0%."] and unknown == ["F7"]


def test_figures_in_verified_filing_quotes_are_supported():
    from equity_agents.agents.worker import verify
    from equity_agents.numbers import unsupported_numbers
    from equity_agents.state import Finding, Source

    passage = (
        '<passage ref="filing:JNJ-10-K-FY2023::s::1">Worldwide sales increased 6.5% to '
        "$85,159 million, driven by Innovative Medicine.</passage>"
    )

    def finding(quote):
        return Finding(
            claim="Sales growth per management",
            source=Source(kind="filing", ref="filing:JNJ-10-K-FY2023::s::1", quote=quote),
        )

    real = finding("Worldwide sales increased 6.5% to $85,159 million")
    fake = finding("Worldwide sales increased 9.1% to $90,000 million")
    verify([real, fake], {"filing:JNJ-10-K-FY2023::s::1"}, None, passage)
    assert real.source.quote_verified and not fake.source.quote_verified
    memo = "Sales rose 6.5% to $85.2 billion [F1]."
    assert unsupported_numbers(memo, [real]) == []
    assert len(unsupported_numbers(memo, [fake])) == 2  # an unverified quote supports nothing
    assert len(unsupported_numbers("Sales rose 7.5% [F1].", [real])) == 1
