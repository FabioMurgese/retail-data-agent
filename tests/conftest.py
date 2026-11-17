"""Shared fixtures."""

from __future__ import annotations

import shutil

import pytest
from fakes import ROOT

from retail_agent.config import Settings


@pytest.fixture
def settings(tmp_path) -> Settings:
    shutil.copytree(ROOT / "config", tmp_path / "config")
    return Settings(
        _env_file=None,
        config_dir=tmp_path / "config",
        golden_dir=ROOT / "golden" / "trios",
        cache_dir=tmp_path / "cache",
        data_dir=tmp_path / "data",
        reports_dir=tmp_path / "reports",
        logs_dir=tmp_path / "logs",
        golden_candidates_dir=tmp_path / "candidates",
        golden_review_dir=tmp_path / "review",
        golden_rejected_dir=tmp_path / "rejected",
        bq_max_rows=100,
    )
