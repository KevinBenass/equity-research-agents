from __future__ import annotations

import pytest
from fakes import PASSAGES, ScriptedBackend, write_companyfacts

from equity_agents.config import load_settings
from equity_agents.memory.store import MemoryStore
from equity_agents.runner import make_deps
from equity_agents.tools.filings import static_filings_tool
from equity_agents.tools.xbrl import FinancialsDB


@pytest.fixture
def settings(tmp_path):
    write_companyfacts(tmp_path / "xbrl")
    return load_settings(
        overrides={
            "tools.xbrl_cache_dir": str(tmp_path / "xbrl"),
            "paths.memory_db": str(tmp_path / "memory.sqlite"),
            "paths.checkpoints_db": str(tmp_path / "checkpoints.sqlite"),
            "paths.runs_dir": str(tmp_path / "runs"),
            "paths.eval_dir": str(tmp_path / "evals"),
        }
    )


@pytest.fixture
def db(settings):
    return FinancialsDB(settings.resolve(settings.tools.xbrl_cache_dir), settings.universe.tickers)


@pytest.fixture
def make(settings, db):
    """Factory for dependencies wired to the scripted backend and offline tools."""

    def _make(run_id="r1", memory=None, settings_override=None, **backend_kwargs):
        backend = ScriptedBackend(**backend_kwargs)
        deps = make_deps(
            settings_override or settings,
            run_id,
            backend=backend,
            filings_tool=static_filings_tool(PASSAGES),
            memory=memory or MemoryStore(settings.resolve(settings.paths.memory_db)),
            db=db,
        )
        return deps, backend

    return _make
