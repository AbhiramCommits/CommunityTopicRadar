"""Tests for configuration and portable database helpers."""

from __future__ import annotations

from datetime import UTC, datetime

import pytest
from sqlalchemy import func, select

from ctr.config import Settings, load_settings
from ctr.db import get_engine, init_db, session_scope, upsert_many
from ctr.models import Post, Tag


def test_load_settings_defaults(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("STACKEXCHANGE_SITE", "stats")
    monkeypatch.setenv("SCRAPE_PAGES", "3")
    monkeypatch.setenv("DATABASE_URL", "sqlite:///example.db")
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    settings = load_settings()
    assert settings.site == "stats"
    assert settings.pages == 3
    assert settings.database_url == "sqlite:///example.db"
    assert settings.openai_api_key is None
    assert settings.openai_model == "gpt-4o-mini"


def test_load_settings_rejects_bad_page_count(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("SCRAPE_PAGES", "zero")
    with pytest.raises(ValueError):
        load_settings()


def test_settings_paths(tmp_path) -> None:
    settings = Settings(data_dir=tmp_path / "data")
    assert settings.raw_dir == tmp_path / "data" / "raw"
    assert settings.cache_dir == tmp_path / "data" / "cache"
    assert settings.sample_dir == tmp_path / "data" / "sample"


def test_init_and_upsert_idempotent(memory_engine) -> None:
    init_db(memory_engine)
    now = datetime.now(UTC)
    post = {
        "id": 1,
        "title": "Hello",
        "body_text": "Body",
        "author_id": 2,
        "author_name": "A",
        "created_at": now,
        "score": 1,
        "view_count": 10,
        "answer_count": 1,
        "is_answered": True,
        "accepted_answer_id": None,
        "url": "https://example.test/q/1",
        "source_site": "stats",
    }
    with session_scope(memory_engine) as session:
        assert upsert_many(session, Post, [post, post], ["id"]) == 1
        assert upsert_many(session, Tag, [{"name": "a"}, {"name": "a"}], ["name"]) == 1
        updated = dict(post, score=5)
        upsert_many(session, Post, [updated], ["id"])
        assert session.scalar(select(Post.score).where(Post.id == 1)) == 5
        assert session.scalar(select(func.count()).select_from(Tag)) == 1


def test_session_scope_rolls_back(memory_engine) -> None:
    with pytest.raises(RuntimeError), session_scope(memory_engine) as session:
        session.add(Tag(name="temporary"))
        raise RuntimeError("boom")
    with session_scope(memory_engine) as session:
        assert session.scalar(select(func.count()).select_from(Tag)) == 0


def test_get_engine(tmp_path) -> None:
    engine = get_engine(f"sqlite:///{tmp_path / 'x.db'}")
    assert engine.dialect.name == "sqlite"
