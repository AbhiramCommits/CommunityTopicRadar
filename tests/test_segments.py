"""Tests for author segments and human labels."""

from __future__ import annotations

import numpy as np
import pandas as pd

from ctr.analysis.segments import (
    FEATURE_COLUMNS,
    build_author_features,
    cluster_authors,
    label_segments,
)


def _blob_frame() -> pd.DataFrame:
    rng = np.random.default_rng(7)
    centers = np.array(
        [
            [10.0, 0.0, 0.0, 1.0, 100.0, 5.0, 0.0, 100.0],
            [1.0, 30.0, 5.0, 2.0, 100.0, 20.0, 30.0, 50.0],
            [2.0, 80.0, 20.0, 5.0, 500.0, 100.0, 40.0, 2.0],
        ]
    )
    rows = []
    author_id = 1
    for center in centers:
        for _ in range(30):
            values = center + rng.normal(0, 0.1, size=len(center))
            record = {"author_id": author_id, **dict(zip(FEATURE_COLUMNS, values, strict=False))}
            rows.append(record)
            author_id += 1
    return pd.DataFrame(rows)


def test_three_blobs_recover_three_clusters() -> None:
    result, metrics = cluster_authors(_blob_frame())
    assert metrics["kmeans_k"] == 3
    assert metrics["kmeans_silhouette"] is not None
    assert metrics["kmeans_silhouette"] > 0.5
    assert result["segment"].nunique() >= 2


def test_label_segments_champion_is_high_answer_centroid() -> None:
    centroids = pd.DataFrame(
        [
            {
                "posts_asked": 1.0,
                "answers_given": 100.0,
                "accepted_answers": 20.0,
                "avg_score": 2.0,
                "median_views": 100.0,
                "active_days": 80.0,
                "answer_to_ask_ratio": 100.0,
                "recency_days": 2.0,
            },
            {
                "posts_asked": 15.0,
                "answers_given": 1.0,
                "accepted_answers": 0.0,
                "avg_score": 1.0,
                "median_views": 80.0,
                "active_days": 5.0,
                "answer_to_ask_ratio": 0.1,
                "recency_days": 90.0,
            },
        ],
        index=[4, 7],
    )
    labels = label_segments(centroids)
    assert labels == ["champion", "asker"]


def test_build_author_features() -> None:
    posts = pd.DataFrame(
        [
            {
                "id": 1,
                "author_id": 1,
                "created_at": pd.Timestamp("2025-01-01", tz="UTC"),
                "score": 2,
                "view_count": 100,
            },
            {
                "id": 2,
                "author_id": 1,
                "created_at": pd.Timestamp("2025-01-03", tz="UTC"),
                "score": 4,
                "view_count": 300,
            },
            {
                "id": 3,
                "author_id": 2,
                "created_at": pd.Timestamp("2025-01-04", tz="UTC"),
                "score": 1,
                "view_count": 50,
            },
        ]
    )
    answers = pd.DataFrame(
        [
            {
                "id": 10,
                "author_id": 1,
                "created_at": pd.Timestamp("2025-01-02", tz="UTC"),
                "is_accepted": True,
            },
            {
                "id": 11,
                "author_id": 2,
                "created_at": pd.Timestamp("2025-01-06", tz="UTC"),
                "is_accepted": False,
            },
        ]
    )
    features = build_author_features(posts, answers)
    author_one = features.set_index("author_id").loc[1]
    assert author_one["posts_asked"] == 2
    assert author_one["answers_given"] == 1
    assert author_one["accepted_answers"] == 1
    assert author_one["active_days"] == 3
    assert author_one["answer_to_ask_ratio"] == 0.5


def test_build_author_features_empty() -> None:
    assert build_author_features(pd.DataFrame(), pd.DataFrame()).empty
