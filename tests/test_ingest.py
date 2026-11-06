"""Tests for the end-to-end ingestion pipeline."""

from __future__ import annotations

from pathlib import Path

from sqlalchemy import func, select
from tests.conftest import SAMPLE_POSTS

from ctr.db import session_scope
from ctr.ingest import run_ingest
from ctr.models import Answer, Author, EngagementDaily, Post, PostTag, Tag


def test_ingest_sample_and_idempotence(settings, memory_engine) -> None:
    first = run_ingest(
        settings,
        offline=True,
        sample_path=SAMPLE_POSTS,
        engine=memory_engine,
        write_stats=False,
    )
    assert first["posts"] == 200
    assert first["answers"] == 184
    assert first["authors"] == 210
    assert first["tags"] == 282
    assert first["post_tags"] == 619
    assert first["engagement_daily"] == 60
    assert first["source"] == "sample"

    second = run_ingest(
        settings,
        offline=True,
        sample_path=SAMPLE_POSTS,
        engine=memory_engine,
        write_stats=False,
    )
    assert second["posts"] == first["posts"]
    assert second["authors"] == first["authors"]
    assert second["post_tags"] == first["post_tags"]
    with session_scope(memory_engine) as session:
        assert session.scalar(select(func.count()).select_from(Post)) == 200
        assert session.scalar(select(func.count()).select_from(Answer)) == 184
        assert session.scalar(select(func.count()).select_from(Author)) == 210
        assert session.scalar(select(func.count()).select_from(Tag)) == 282
        assert session.scalar(select(func.count()).select_from(PostTag)) == 619
        assert session.scalar(select(func.count()).select_from(EngagementDaily)) == 60


def test_ingest_offline_falls_back_to_sample(settings, memory_engine) -> None:
    sample_dir = settings.data_dir / "sample"
    sample_dir.mkdir(parents=True)
    for source in (SAMPLE_POSTS, SAMPLE_POSTS.with_name("answers_sample.json")):
        (sample_dir / source.name).write_bytes(source.read_bytes())
    stats = run_ingest(settings, offline=True, engine=memory_engine, write_stats=False)
    assert stats["source"] == "sample"
    assert stats["posts"] == 200


def test_ingest_writes_stats(settings, memory_engine) -> None:
    stats = run_ingest(
        settings,
        offline=True,
        sample_path=SAMPLE_POSTS,
        engine=memory_engine,
        write_stats=True,
    )
    output = Path(settings.reports_dir) / "ingest_stats.json"
    assert output.exists()
    assert stats["date_min"].startswith("2026-09-02")
