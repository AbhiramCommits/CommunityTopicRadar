"""Tests for trending-term ranking and NMF topic modelling."""

from __future__ import annotations

import pandas as pd

from ctr.analysis.topics import rank_trending, topic_model


def test_planted_spike_ranks_first() -> None:
    rows = []
    for index in range(8):
        rows.append(
            {
                "id": index,
                "title": f"Question {index}",
                "body_text": f"spike uniquecurrent{index}",
                "created_at": pd.Timestamp("2025-03-08", tz="UTC")
                + pd.Timedelta(days=index % 7),
                "tags": [],
            }
        )
        rows.append(
            {
                "id": 100 + index,
                "title": f"Older question {index}",
                "body_text": f"ordinary uniqueprior{index}",
                "created_at": pd.Timestamp("2025-03-01", tz="UTC")
                + pd.Timedelta(days=index % 7),
                "tags": [],
            }
        )
    rows.append(
        {
            "id": 999,
            "title": "Today only",
            "body_text": "today extra",
            "created_at": pd.Timestamp("2025-03-15", tz="UTC"),
            "tags": [],
        }
    )
    frame = pd.DataFrame(rows)
    trending = rank_trending(frame)
    assert trending.iloc[0]["term"] == "spike"
    assert trending.iloc[0]["current_count"] == 8
    assert trending.iloc[0]["prior_count"] == 0
    assert trending.iloc[0]["z_score"] > 0


def test_rank_trending_empty() -> None:
    result = rank_trending(pd.DataFrame())
    assert result.empty
    assert list(result.columns) == [
        "term",
        "current_count",
        "prior_count",
        "delta",
        "pct_change",
        "z_score",
        "kind",
    ]


def test_topic_model_returns_k_topics_with_ten_terms() -> None:
    themes = [
        "regression linear residual variance prediction",
        "survival hazard cox censoring treatment",
        "bayesian posterior prior likelihood sampling",
        "clustering distance centroid silhouette neighbors",
        "forecasting arima seasonal stationarity horizon",
        "neural network gradient embedding training",
    ]
    rows = []
    for index in range(120):
        theme = themes[index % len(themes)]
        rows.append(
            {
                "id": index,
                "title": f"{theme.split()[0]} question {index}",
                "body_text": f"{theme} example{index}",
                "created_at": pd.Timestamp("2025-01-01", tz="UTC")
                + pd.Timedelta(days=index),
                "tags": [],
            }
        )
    frame = pd.DataFrame(rows)
    topics, assignments = topic_model(frame, k=6)
    assert topics["topic"].nunique() == 6
    assert (topics.groupby("topic").size() == 10).all()
    assert len(assignments) == len(frame)
    assert assignments["topic"].between(0, 5).all()


def test_topic_model_empty_frame() -> None:
    topics, assignments = topic_model(pd.DataFrame(), k=3)
    assert topics.empty
    assert assignments.empty
