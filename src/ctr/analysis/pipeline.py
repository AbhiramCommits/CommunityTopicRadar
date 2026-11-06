"""End-to-end analytics pipeline that writes the report artifacts."""

from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sqlalchemy import Engine, func, select
from tabulate import tabulate

from ctr.analysis.anomaly import detect_anomalies
from ctr.analysis.content_gap import gap_score
from ctr.analysis.segments import (
    FEATURE_COLUMNS,
    build_author_features,
    cluster_authors,
)
from ctr.analysis.topics import rank_trending, topic_model
from ctr.config import Settings
from ctr.db import get_engine
from ctr.models import Answer, EngagementDaily, Post, PostTag, Tag

logger = logging.getLogger(__name__)


@dataclass
class AnalysisData:
    """Frames loaded once from the warehouse for all analytics."""

    posts: pd.DataFrame
    answers: pd.DataFrame
    daily: pd.DataFrame


def load_analysis_data(engine: Engine) -> AnalysisData:
    """Load posts, tags, first-answer latency, answers, and daily engagement."""
    with engine.connect() as connection:
        posts = pd.read_sql_query(
            select(
                Post.id,
                Post.title,
                Post.body_text,
                Post.author_id,
                Post.author_name,
                Post.created_at,
                Post.score,
                Post.view_count,
                Post.answer_count,
                Post.is_answered,
                Post.accepted_answer_id,
                Post.url,
                Post.source_site,
            ),
            connection,
        )
        tag_rows = pd.read_sql_query(
            select(PostTag.post_id, Tag.name).join(Tag, PostTag.tag_id == Tag.id),
            connection,
        )
        first_answers = pd.read_sql_query(
            select(
                Answer.question_id,
                func.min(Answer.created_at).label("first_answer_at"),
            ).group_by(Answer.question_id),
            connection,
        )
        answers = pd.read_sql_query(
            select(
                Answer.id,
                Answer.question_id,
                Answer.author_id,
                Answer.author_name,
                Answer.created_at,
                Answer.score,
                Answer.is_accepted,
            ),
            connection,
        )
        daily = pd.read_sql_query(
            select(
                EngagementDaily.day,
                EngagementDaily.posts_created,
                EngagementDaily.answers_created,
                EngagementDaily.unique_authors,
                EngagementDaily.answered_count,
                EngagementDaily.total_views,
            ),
            connection,
        )

    posts["created_at"] = pd.to_datetime(posts["created_at"], utc=True, errors="coerce")
    posts["is_answered"] = posts["is_answered"].fillna(False).astype(bool)
    if not tag_rows.empty:
        grouped_tags = tag_rows.groupby("post_id")["name"].apply(list)
        posts["tags"] = posts["id"].map(grouped_tags)
        posts["tags"] = posts["tags"].apply(lambda value: value if isinstance(value, list) else [])
    else:
        posts["tags"] = [[] for _ in range(len(posts))]
    if not first_answers.empty:
        first_answers["first_answer_at"] = pd.to_datetime(
            first_answers["first_answer_at"], utc=True, errors="coerce"
        )
        posts = posts.merge(
            first_answers, left_on="id", right_on="question_id", how="left"
        )
        posts = posts.drop(columns=["question_id"], errors="ignore")
    else:
        posts["first_answer_at"] = pd.NaT

    answers["created_at"] = pd.to_datetime(answers["created_at"], utc=True, errors="coerce")
    answers["is_accepted"] = answers["is_accepted"].fillna(False).astype(bool)
    daily["day"] = pd.to_datetime(daily["day"], utc=True, errors="coerce")
    return AnalysisData(posts=posts, answers=answers, daily=daily)


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, list | tuple):
        return [_json_safe(item) for item in value]
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value) if np.isfinite(value) else None
    if isinstance(value, np.bool_):
        return bool(value)
    if isinstance(value, float) and not np.isfinite(value):
        return None
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    return value


def _records(frame: pd.DataFrame, limit: int | None = None) -> list[dict[str, Any]]:
    selected = frame.head(limit) if limit else frame
    return [
        {key: _json_safe(value) for key, value in record.items()}
        for record in selected.to_dict("records")
    ]


def _write_segment_profile(segments: pd.DataFrame, path: Path) -> None:
    rows = []
    for segment, group in segments.groupby("segment", sort=True):
        row: dict[str, Any] = {"segment": segment, "authors": len(group)}
        for column in FEATURE_COLUMNS:
            row[column] = round(float(group[column].mean()), 2)
        rows.append(row)
    profile = pd.DataFrame(rows)
    path.write_text(
        "# Community segment profile\n\n"
        "Mean feature values per K-Means segment (features are standardized "
        "before clustering; values below are in original units).\n\n"
        + tabulate(profile, headers="keys", tablefmt="github", showindex=False)
        + "\n",
        encoding="utf-8",
    )


def _summary(
    data: AnalysisData,
    chosen_k: int,
    trending: pd.DataFrame,
    gaps: pd.DataFrame,
    segment_metrics: dict[str, Any],
    anomaly_metrics: dict[str, Any],
) -> str:
    date_min = data.posts["created_at"].min()
    date_max = data.posts["created_at"].max()
    trending_text = "\n".join(
        f"{record['term']} ({record['pct_change']:+.1f}%)"
        for record in _records(trending, 5)
    )
    gaps_text = "\n".join(
        f"{record['topic']} (answer {record['answer_rate']:.0%}, gap {record['gap_score']:.3f})"
        for record in _records(gaps, 5)
    )
    kmeans_silhouette = segment_metrics.get("kmeans_silhouette")
    dbscan_silhouette = segment_metrics.get("dbscan_silhouette")
    rows = [
        ["Posts analyzed", f"{len(data.posts):,}"],
        ["Answers analyzed", f"{len(data.answers):,}"],
        [
            "Date range",
            f"{date_min.date().isoformat()} to {date_max.date().isoformat()}",
        ],
        ["Chosen NMF k", chosen_k],
        ["Top 5 trending terms", trending_text],
        ["Top 5 content gaps", gaps_text],
        [
            "K-Means k + silhouette",
            f"{segment_metrics.get('kmeans_k')} / "
            f"{kmeans_silhouette:.3f}" if kmeans_silhouette is not None else "n/a",
        ],
        [
            "DBSCAN silhouette",
            f"{dbscan_silhouette:.3f}" if dbscan_silhouette is not None else "n/a",
        ],
        [
            "Anomalies found",
            f"{anomaly_metrics.get('anomaly_count')} rows over "
            f"{anomaly_metrics.get('anomaly_days')} days",
        ],
    ]
    return tabulate(rows, headers=["metric", "value"], tablefmt="github")


def run_analysis(settings: Settings) -> str:
    """Run every analytics module on the warehouse and write report artifacts."""
    started = time.perf_counter()
    reports = settings.reports_dir
    reports.mkdir(parents=True, exist_ok=True)
    engine = get_engine(settings.database_url)
    data = load_analysis_data(engine)
    if data.posts.empty:
        raise RuntimeError("No posts found; run `python -m ctr.cli ingest` first")
    logger.info("Loaded %d posts and %d answers", len(data.posts), len(data.answers))

    trending = rank_trending(data.posts, window_days=7)
    trending.head(100).to_csv(reports / "trending_topics.csv", index=False)
    topics, assignments = topic_model(data.posts)
    chosen_k = int(topics.attrs.get("k", 0))
    topics.to_csv(reports / "topics.csv", index=False)
    assignments.to_csv(reports / "topic_assignments.csv", index=False)

    min_posts = 10 if len(data.posts) >= 1000 else 2
    gaps = gap_score(data.posts, min_posts=min_posts)
    gaps.head(20).to_csv(reports / "content_gaps.csv", index=False)

    author_features = build_author_features(data.posts, data.answers)
    segments, segment_metrics = cluster_authors(author_features)
    segments.to_csv(reports / "segments.csv", index=False)
    _write_segment_profile(segments, reports / "segment_profile.md")

    anomalies, anomaly_metrics = detect_anomalies(data.daily)
    anomalies.to_csv(reports / "anomalies.csv", index=False)

    segment_sizes = (
        segments["segment"].value_counts().sort_index().to_dict() if not segments.empty else {}
    )
    top_terms = {
        int(topic): group.sort_values("rank")["term"].head(5).tolist()
        for topic, group in topics.groupby("topic")
    } if not topics.empty else {}
    metrics: dict[str, Any] = {
        "generated_at": pd.Timestamp.now(tz="UTC").isoformat(),
        "posts_analyzed": len(data.posts),
        "answers_analyzed": len(data.answers),
        "authors_analyzed": len(author_features),
        "tags_analyzed": int(data.posts["tags"].explode().dropna().nunique()),
        "days_of_engagement": int(data.daily["day"].nunique()),
        "date_min": data.posts["created_at"].min().isoformat(),
        "date_max": data.posts["created_at"].max().isoformat(),
        "nmf_k": chosen_k,
        "nmf_reconstruction_error": _json_safe(
            topics.attrs.get("reconstruction_error")
        ),
        "topic_top_terms": top_terms,
        "trending_top_terms": _records(trending, 5),
        "content_gaps_top": _records(gaps, 5),
        "kmeans_k": segment_metrics.get("kmeans_k"),
        "kmeans_silhouette": _json_safe(segment_metrics.get("kmeans_silhouette")),
        "dbscan_silhouette": _json_safe(segment_metrics.get("dbscan_silhouette")),
        "dbscan_clusters": segment_metrics.get("dbscan_clusters"),
        "segment_sizes": {str(key): int(value) for key, value in segment_sizes.items()},
        "anomaly_count": anomaly_metrics.get("anomaly_count"),
        "anomaly_days": anomaly_metrics.get("anomaly_days"),
        "robust_z_days": anomaly_metrics.get("robust_z_days"),
        "isolation_forest_days": anomaly_metrics.get("isolation_forest_days"),
        "iforest_precision": _json_safe(anomaly_metrics.get("iforest_precision")),
        "iforest_recall": _json_safe(anomaly_metrics.get("iforest_recall")),
        "analysis_seconds": round(time.perf_counter() - started, 2),
    }
    (reports / "metrics.json").write_text(
        json.dumps(_json_safe(metrics), indent=2), encoding="utf-8"
    )
    summary = _summary(data, chosen_k, trending, gaps, segment_metrics, anomaly_metrics)
    logger.info("Wrote analytics artifacts to %s in %.1fs", reports, metrics["analysis_seconds"])
    return summary
