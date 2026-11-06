"""Content-gap scoring for topic-level demand and supply."""

from __future__ import annotations

import logging

import pandas as pd

logger = logging.getLogger(__name__)

GAP_COLUMNS = [
    "topic",
    "posts",
    "answer_rate",
    "median_views",
    "gap_score",
    "example_post_url",
    "accepted_answer_rate",
    "median_time_to_first_answer_hours",
    "demand_percentile",
]


def _empty_gaps() -> pd.DataFrame:
    return pd.DataFrame(columns=GAP_COLUMNS)


def gap_score(df: pd.DataFrame, min_posts: int = 5) -> pd.DataFrame:
    """Rank tags by demand for answers against the supply of answered posts.

    Demand combines the percentile rank of post volume with the percentile
    rank of median views. Supply is the answer rate. The gap score is
    ``demand_percentile * (1 - answer_rate)``.
    """
    if df.empty or "tags" not in df.columns:
        return _empty_gaps()
    working = df.copy()
    working["created_at"] = pd.to_datetime(working["created_at"], utc=True, errors="coerce")
    working["is_answered"] = working["is_answered"].fillna(False).astype(bool)
    working["has_accepted"] = working["accepted_answer_id"].notna()
    if "first_answer_at" in working.columns:
        first_answer = pd.to_datetime(working["first_answer_at"], utc=True, errors="coerce")
        working["_time_to_answer_hours"] = (
            first_answer - working["created_at"]
        ).dt.total_seconds() / 3600.0
    else:
        working["_time_to_answer_hours"] = float("nan")

    exploded = working.explode("tags").dropna(subset=["tags"]).copy()
    if exploded.empty:
        return _empty_gaps()
    exploded["tags"] = exploded["tags"].astype(str)

    grouped = exploded.groupby("tags", dropna=False)
    summary = grouped.agg(
        posts=("id", "count"),
        answer_rate=("is_answered", "mean"),
        median_views=("view_count", "median"),
        accepted_answer_rate=("has_accepted", "mean"),
        median_time_to_first_answer_hours=("_time_to_answer_hours", "median"),
    ).reset_index()
    summary = summary.rename(columns={"tags": "topic"})
    summary = summary[summary["posts"] >= min_posts].copy()
    if summary.empty:
        return _empty_gaps()

    unanswered = exploded[~exploded["is_answered"]].sort_values("view_count", ascending=False)
    fallback = exploded.sort_values("view_count", ascending=False)
    examples: dict[str, str] = {}
    for topic in summary["topic"]:
        candidate = unanswered[unanswered["tags"] == topic]
        if candidate.empty:
            candidate = fallback[fallback["tags"] == topic]
        examples[str(topic)] = str(candidate.iloc[0]["url"]) if not candidate.empty else ""
    summary["example_post_url"] = summary["topic"].map(examples)

    volume_percentile = summary["posts"].rank(pct=True, method="average")
    views_percentile = summary["median_views"].rank(pct=True, method="average")
    summary["demand_percentile"] = (volume_percentile + views_percentile) / 2.0
    summary["gap_score"] = summary["demand_percentile"] * (1.0 - summary["answer_rate"])
    summary = summary.sort_values(
        ["gap_score", "median_views", "posts"], ascending=False
    ).reset_index(drop=True)
    logger.info("Scored %d tag topics for content gaps", len(summary))
    return summary[GAP_COLUMNS]
