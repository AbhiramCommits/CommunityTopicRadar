"""Tests for the Power BI star-schema export."""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

import pandas as pd
from sqlalchemy import create_engine

from ctr import export
from ctr.analysis.pipeline import AnalysisData
from ctr.db import init_db, session_scope
from ctr.models import Author

ROOT = Path(__file__).resolve().parents[1]


def test_committed_star_schema_columns_and_integrity() -> None:
    exports = ROOT / "exports"
    fact_post = pd.read_csv(exports / "fact_post.csv")
    dim_topic = pd.read_csv(exports / "dim_topic.csv")
    dim_date = pd.read_csv(exports / "dim_date.csv")
    assert {
        "post_id",
        "topic_id",
        "author_id",
        "date_key",
        "score",
        "view_count",
        "answer_count",
        "is_answered",
    } <= set(fact_post.columns)
    assert set(fact_post["topic_id"]).issubset(set(dim_topic["topic_id"]))
    assert set(fact_post["date_key"]).issubset(set(dim_date["date_key"]))
    assert dim_date["date_key"].is_unique
    assert (exports / "powerbi_measures.md").exists()
    assert (ROOT / "sql" / "views.sql").exists()


def test_run_export_writes_all_files(settings, monkeypatch) -> None:
    settings.reports_dir.mkdir(parents=True, exist_ok=True)
    engine = create_engine(settings.database_url)
    init_db(engine)
    with session_scope(engine) as session:
        for author_id in (1, 2):
            session.add(
                Author(
                    id=author_id,
                    name=f"Author {author_id}",
                    reputation=10,
                    first_seen_at=datetime(2025, 1, 1, tzinfo=UTC),
                    last_seen_at=datetime(2025, 1, 3, tzinfo=UTC),
                )
            )
    pd.DataFrame(
        [
            {"topic": 0, "rank": 1, "term": "alpha", "weight": 1.0},
            {"topic": 1, "rank": 1, "term": "beta", "weight": 1.0},
        ]
    ).to_csv(settings.reports_dir / "topics.csv", index=False)
    pd.DataFrame(
        [
            {"id": 1, "topic": 0, "weight": 0.9},
            {"id": 2, "topic": 1, "weight": 0.8},
            {"id": 3, "topic": 0, "weight": 0.7},
        ]
    ).to_csv(settings.reports_dir / "topic_assignments.csv", index=False)
    pd.DataFrame(
        [
            {
                "author_id": 1,
                "segment": "asker",
                "cluster": 0,
                "posts_asked": 2,
                "answers_given": 0,
                "accepted_answers": 0,
                "active_days": 2,
                "pca_x": 0.1,
                "pca_y": 0.2,
            },
            {
                "author_id": 2,
                "segment": "answerer",
                "cluster": 1,
                "posts_asked": 1,
                "answers_given": 4,
                "accepted_answers": 1,
                "active_days": 5,
                "pca_x": -0.2,
                "pca_y": 0.4,
            },
        ]
    ).to_csv(settings.reports_dir / "segments.csv", index=False)
    pd.DataFrame(
        [
            {
                "day": "2025-01-02",
                "metric": "posts_created",
                "value": 10,
                "expected": 2,
                "deviation": 8,
                "methods": "robust_z",
                "direction": "spike",
            }
        ]
    ).to_csv(settings.reports_dir / "anomalies.csv", index=False)

    posts = pd.DataFrame(
        [
            {
                "id": 1,
                "title": "One",
                "body_text": "b",
                "author_id": 1,
                "author_name": "Author 1",
                "created_at": pd.Timestamp("2025-01-01", tz="UTC"),
                "score": 1,
                "view_count": 10,
                "answer_count": 1,
                "is_answered": True,
                "accepted_answer_id": 9,
                "url": "https://example.test/1",
                "source_site": "stats",
                "tags": ["alpha"],
                "first_answer_at": pd.Timestamp("2025-01-02", tz="UTC"),
            },
            {
                "id": 2,
                "title": "Two",
                "body_text": "b",
                "author_id": 2,
                "author_name": "Author 2",
                "created_at": pd.Timestamp("2025-01-02", tz="UTC"),
                "score": 2,
                "view_count": 20,
                "answer_count": 0,
                "is_answered": False,
                "accepted_answer_id": None,
                "url": "https://example.test/2",
                "source_site": "stats",
                "tags": ["beta"],
                "first_answer_at": pd.NaT,
            },
            {
                "id": 3,
                "title": "Three",
                "body_text": "b",
                "author_id": 1,
                "author_name": "Author 1",
                "created_at": pd.Timestamp("2025-01-03", tz="UTC"),
                "score": 0,
                "view_count": 5,
                "answer_count": 0,
                "is_answered": False,
                "accepted_answer_id": None,
                "url": "https://example.test/3",
                "source_site": "stats",
                "tags": ["alpha"],
                "first_answer_at": pd.NaT,
            },
        ]
    )
    daily = pd.DataFrame(
        [
            {
                "day": pd.Timestamp("2025-01-01", tz="UTC"),
                "posts_created": 1,
                "answers_created": 1,
                "unique_authors": 1,
                "answered_count": 1,
                "total_views": 10,
            },
            {
                "day": pd.Timestamp("2025-01-02", tz="UTC"),
                "posts_created": 10,
                "answers_created": 2,
                "unique_authors": 2,
                "answered_count": 0,
                "total_views": 200,
            },
        ]
    )
    monkeypatch.setattr(
        export,
        "load_analysis_data",
        lambda engine: AnalysisData(posts=posts, answers=pd.DataFrame(), daily=daily),
    )
    outputs = export.run_export(settings)
    assert len(outputs) >= 8
    for name in export.STAR_FILES:
        assert (settings.exports_dir / name).exists()
    fact = pd.read_csv(settings.exports_dir / "fact_post.csv")
    topic = pd.read_csv(settings.exports_dir / "dim_topic.csv")
    date_dim = pd.read_csv(settings.exports_dir / "dim_date.csv")
    assert set(fact["topic_id"]).issubset(set(topic["topic_id"]))
    assert set(fact["date_key"]).issubset(set(date_dim["date_key"]))
    assert date_dim["date_key"].is_unique
    assert len(date_dim) == 3
