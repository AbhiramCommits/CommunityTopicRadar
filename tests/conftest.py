"""Shared pytest fixtures."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import pytest
from sqlalchemy import create_engine
from sqlalchemy.pool import StaticPool

from ctr.config import Settings, load_settings
from ctr.db import init_db

ROOT = Path(__file__).resolve().parents[1]
SAMPLE_POSTS = ROOT / "data" / "sample" / "posts_sample.json"
FIXTURES = Path(__file__).resolve().parent / "fixtures"


@pytest.fixture()
def settings(tmp_path: Path) -> Settings:
    """Isolated settings for tests that write files or databases."""
    return replace(
        load_settings(),
        database_url=f"sqlite:///{tmp_path / 'test.db'}",
        openai_api_key=None,
        data_dir=tmp_path / "data",
        reports_dir=tmp_path / "reports",
        exports_dir=tmp_path / "exports",
        pages=1,
    )


@pytest.fixture()
def memory_engine():
    """A shared in-memory SQLite engine that survives multiple sessions."""
    engine = create_engine(
        "sqlite://",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    init_db(engine)
    yield engine
    engine.dispose()
