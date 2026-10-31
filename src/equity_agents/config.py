"""Typed configuration loaded from YAML, with dotted-key overrides."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = ROOT / "configs" / "default.yaml"


class Price(BaseModel):
    input: float
    output: float


class Universe(BaseModel):
    tickers: list[str]
    fiscal_years: list[int]


class Models(BaseModel):
    planner: str
    filings_researcher: str
    quant_analyst: str
    writer: str
    critic: str
    memory_extractor: str
    prices: dict[str, Price]


class Orchestration(BaseModel):
    max_tasks: int = 6
    max_worker_steps: int = 8
    max_revisions: int = 2
    max_research_rounds: int = 2
    human_review: bool = False


class Context(BaseModel):
    worker_budget_tokens: int = 12_000
    tool_result_max_chars: int = 4_000
    keep_recent_tool_results: int = 3
    memory_facts_in_prompt: int = 20
    episodes_in_prompt: int = 2


class Tools(BaseModel):
    filings_api_url: str = "http://localhost:8000"
    sec_user_agent: str = "equity-research-agents"
    xbrl_cache_dir: Path = Path("data/xbrl")
    sql_row_limit: int = 200


class Paths(BaseModel):
    memory_db: Path = Path("data/memory.sqlite")
    checkpoints_db: Path = Path("data/checkpoints.sqlite")
    runs_dir: Path = Path("runs")
    eval_dir: Path = Path("evals")


class Settings(BaseModel):
    universe: Universe
    models: Models
    orchestration: Orchestration
    context: Context
    tools: Tools
    paths: Paths

    def resolve(self, p: Path) -> Path:
        return p if p.is_absolute() else ROOT / p

    @property
    def sec_user_agent(self) -> str:
        return os.environ.get("SEC_USER_AGENT", self.tools.sec_user_agent)

    @property
    def filings_api_url(self) -> str:
        return os.environ.get("FILINGS_API_URL", self.tools.filings_api_url)


def load_settings(
    path: Path | str | None = None, overrides: dict[str, Any] | None = None
) -> Settings:
    raw = yaml.safe_load(Path(path or DEFAULT_CONFIG).read_text())
    for dotted, value in (overrides or {}).items():
        node = raw
        *parents, leaf = dotted.split(".")
        for key in parents:
            node = node[key]
        if leaf not in node:
            raise KeyError(f"Unknown config key: {dotted}")
        node[leaf] = value
    return Settings.model_validate(raw)
