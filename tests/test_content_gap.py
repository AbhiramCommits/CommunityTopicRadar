"""Tests for content-gap scoring."""

from __future__ import annotations

import pandas as pd

from ctr.analysis.content_gap import gap_score


def test_unanswered_high_view_topic_outranks_answered_topic() -> None:
    rows = []
    for index in range(10):
        rows.append(
            {
                "id": index,
                "title": "Rare topic",
                "body_text": "",
                "created_at": pd.Timestamp("2025-01-01", tz="UTC"),
                "score": 1,
                "view_count": 1000 + index,
                "is_answered": False,
                "accepted_answer_id": None,
                "url": f"https://example.test/rare/{index}",
                "tags": ["rare-topic"],
                "first_answer_at": pd.NaT,
            }
        )
        rows.append(
            {
                "id": 100 + index,
                "title": "Answered topic",
                "body_text": "",
                "created_at": pd.Timestamp("2025-01-01", tz="UTC"),
                "score": 1,
                "view_count": 100 + index,
                "is_answered": True,
                "accepted_answer_id": 999,
                "url": f"https://example.test/answered/{index}",
                "tags": ["answered-topic"],
                "first_answer_at": pd.Timestamp("2025-01-02", tz="UTC"),
            }
        )
    result = gap_score(pd.DataFrame(rows), min_posts=2)
    assert result.iloc[0]["topic"] == "rare-topic"
    assert result.iloc[0]["answer_rate"] == 0.0
    assert result.iloc[0]["gap_score"] > result.iloc[1]["gap_score"]
    assert result.iloc[0]["example_post_url"].startswith("https://example.test/rare/")


def test_gap_score_empty() -> None:
    assert gap_score(pd.DataFrame()).empty
