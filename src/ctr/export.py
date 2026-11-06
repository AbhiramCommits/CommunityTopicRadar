"""Power BI-ready star-schema export and SQL view documentation."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import pandas as pd
from sqlalchemy import select

from ctr.analysis.pipeline import load_analysis_data
from ctr.config import Settings
from ctr.db import get_engine
from ctr.models import Author

logger = logging.getLogger(__name__)

STAR_FILES = [
    "dim_topic.csv",
    "dim_author.csv",
    "dim_date.csv",
    "fact_post.csv",
    "fact_engagement_daily.csv",
    "fact_anomaly.csv",
]


def _date_key(values: pd.Series) -> pd.Series:
    return pd.to_datetime(values, utc=True).dt.strftime("%Y%m%d").astype(int)


def _dim_date(dates: pd.Series) -> pd.DataFrame:
    converted = pd.to_datetime(dates, utc=True, errors="coerce").dropna()
    if converted.empty:
        frame = pd.DataFrame(columns=["date"])
    else:
        normalized = converted.dt.tz_convert("UTC").dt.normalize().dt.tz_localize(None)
        full_range = pd.date_range(normalized.min(), normalized.max(), freq="D")
        frame = pd.DataFrame({"date": full_range})
    frame["date_key"] = frame["date"].dt.strftime("%Y%m%d").astype(int)
    frame["year"] = frame["date"].dt.year
    frame["quarter"] = frame["date"].dt.quarter
    frame["month"] = frame["date"].dt.month
    frame["month_name"] = frame["date"].dt.strftime("%B")
    frame["week"] = frame["date"].dt.isocalendar().week.astype(int)
    frame["day_of_week"] = frame["date"].dt.day_name()
    frame["is_weekend"] = frame["date"].dt.dayofweek >= 5
    return frame[
        [
            "date_key",
            "date",
            "year",
            "quarter",
            "month",
            "month_name",
            "week",
            "day_of_week",
            "is_weekend",
        ]
    ]


def _topic_dimension(
    posts: pd.DataFrame, assignments: pd.DataFrame, topics: pd.DataFrame
) -> pd.DataFrame:
    merged = posts.merge(assignments[["id", "topic"]], on="id", how="left")
    merged["topic"] = merged["topic"].fillna(-1).astype(int)
    rows: list[dict[str, Any]] = []
    for topic_id, group in merged.groupby("topic"):
        terms = topics[topics["topic"] == topic_id].sort_values("rank")["term"].tolist()
        label = " / ".join(str(term) for term in terms[:3]) if terms else "Unassigned"
        rows.append(
            {
                "topic_id": int(topic_id),
                "topic_label": label,
                "top_terms": ", ".join(str(term) for term in terms[:10]),
                "posts": int(len(group)),
                "answer_rate": float(group["is_answered"].mean()) if len(group) else 0.0,
                "median_views": float(group["view_count"].median()) if len(group) else 0.0,
                "gap_score": 0.0,
            }
        )
    dimension = pd.DataFrame(rows)
    if len(dimension) > 1:
        demand = (
            dimension["posts"].rank(pct=True)
            + dimension["median_views"].rank(pct=True)
        ) / 2.0
        dimension["gap_score"] = demand * (1.0 - dimension["answer_rate"])
    return dimension.sort_values("topic_id").reset_index(drop=True)


def _author_dimension(
    posts: pd.DataFrame, engine: Any, segments: pd.DataFrame
) -> pd.DataFrame:
    with engine.connect() as connection:
        authors = pd.read_sql_query(
            select(
                Author.id,
                Author.name,
                Author.reputation,
                Author.first_seen_at,
                Author.last_seen_at,
            ),
            connection,
        )
    used_ids = posts["author_id"].dropna().astype(int).unique()
    if authors.empty:
        authors = pd.DataFrame(
            columns=["id", "name", "reputation", "first_seen_at", "last_seen_at"]
        )
    known = set(authors["id"].astype(int))
    missing = [
        {
            "id": int(author_id),
            "name": "unknown",
            "reputation": 0,
            "first_seen_at": pd.NaT,
            "last_seen_at": pd.NaT,
        }
        for author_id in used_ids
        if int(author_id) not in known
    ]
    if missing:
        authors = pd.concat([authors, pd.DataFrame(missing)], ignore_index=True)
    authors = authors.rename(columns={"id": "author_id"})
    segment_columns = [
        "author_id",
        "segment",
        "cluster",
        "posts_asked",
        "answers_given",
        "accepted_answers",
        "active_days",
        "pca_x",
        "pca_y",
    ]
    if not segments.empty:
        authors = authors.merge(
            segments[[column for column in segment_columns if column in segments.columns]],
            on="author_id",
            how="left",
        )
    for column in ("segment", "cluster"):
        if column not in authors:
            authors[column] = pd.NA
    authors["segment"] = authors["segment"].fillna("unclassified")
    return authors


def _write_powerbi_measures(path: Path) -> None:
    path.write_text(
        """# Power BI measures

Import the six CSVs in this folder as a star schema. Relate
`fact_post[topic_id]` to `dim_topic[topic_id]`, `fact_post[author_id]` to
`dim_author[author_id]`, and every `date_key` to `dim_date[date_key]`.

```DAX
Monthly Active Authors =
DISTINCTCOUNT ( fact_post[author_id] )

Answer Rate =
DIVIDE (
    CALCULATE ( COUNTROWS ( fact_post ), fact_post[is_answered] = TRUE() ),
    COUNTROWS ( fact_post )
)

WoW Topic Growth =
VAR CurrentWeek = MAX ( dim_date[date] )
VAR CurrentPosts =
    CALCULATE (
        COUNTROWS ( fact_post ),
        FILTER ( dim_date, dim_date[date] > CurrentWeek - 7 && dim_date[date] <= CurrentWeek )
    )
VAR PriorPosts =
    CALCULATE (
        COUNTROWS ( fact_post ),
        FILTER (
            dim_date,
            dim_date[date] > CurrentWeek - 14 && dim_date[date] <= CurrentWeek - 7
        )
    )
RETURN DIVIDE ( CurrentPosts - PriorPosts, PriorPosts )

Content Gap Backlog Size =
COUNTROWS ( FILTER ( dim_topic, dim_topic[gap_score] >= 0.30 ) )
```

`fact_engagement_daily[is_anomaly]` and `fact_anomaly` support the anomaly
page. `dim_author[segment]` supports the community-segment page.
""",
        encoding="utf-8",
    )


def _write_sql_views(path: Path) -> None:
    path.write_text(
        """-- PostgreSQL views equivalent to the Power BI measures.
CREATE OR REPLACE VIEW v_monthly_active_authors AS
SELECT
    DATE_TRUNC('month', created_at)::date AS month,
    COUNT(DISTINCT author_id) AS monthly_active_authors
FROM posts
GROUP BY 1;

CREATE OR REPLACE VIEW v_answer_rate AS
SELECT
    DATE_TRUNC('month', created_at)::date AS month,
    AVG(CASE WHEN is_answered THEN 1.0 ELSE 0.0 END) AS answer_rate
FROM posts
GROUP BY 1;

CREATE OR REPLACE VIEW v_topic_wow_growth AS
WITH weekly AS (
    SELECT
        t.name AS topic,
        DATE_TRUNC('week', p.created_at)::date AS week,
        COUNT(*) AS posts
    FROM posts AS p
    JOIN post_tags AS pt ON pt.post_id = p.id
    JOIN tags AS t ON t.id = pt.tag_id
    GROUP BY 1, 2
),
comparison AS (
    SELECT
        topic,
        week,
        posts,
        LAG(posts) OVER (PARTITION BY topic ORDER BY week) AS prior_posts
    FROM weekly
)
SELECT
    topic,
    week,
    posts,
    prior_posts,
    CASE
        WHEN prior_posts > 0 THEN (posts - prior_posts)::numeric / prior_posts
        ELSE NULL
    END AS wow_growth
FROM comparison;

CREATE OR REPLACE VIEW v_content_gap_backlog AS
WITH tag_stats AS (
    SELECT
        t.name AS topic,
        COUNT(*) AS posts,
        AVG(CASE WHEN p.is_answered THEN 1.0 ELSE 0.0 END) AS answer_rate,
        PERCENTILE_CONT(0.5) WITHIN GROUP (ORDER BY p.view_count) AS median_views
    FROM posts AS p
    JOIN post_tags AS pt ON pt.post_id = p.id
    JOIN tags AS t ON t.id = pt.tag_id
    GROUP BY 1
),
scored AS (
    SELECT
        *,
        (
            PERCENT_RANK() OVER (ORDER BY posts)
            + PERCENT_RANK() OVER (ORDER BY median_views)
        ) / 2.0 AS demand_percentile
    FROM tag_stats
)
SELECT
    topic,
    posts,
    answer_rate,
    median_views,
    demand_percentile * (1 - answer_rate) AS gap_score
FROM scored;
""",
        encoding="utf-8",
    )


def run_export(settings: Settings) -> list[Path]:
    """Write six star-schema CSVs plus DAX and SQL documentation."""
    reports = settings.reports_dir
    required = {
        "topics": reports / "topics.csv",
        "assignments": reports / "topic_assignments.csv",
        "segments": reports / "segments.csv",
        "anomalies": reports / "anomalies.csv",
    }
    for label, path in required.items():
        if not path.exists():
            raise FileNotFoundError(f"Missing {label} artifact {path}; run analyze first")
    topics = pd.read_csv(required["topics"])
    assignments = pd.read_csv(required["assignments"])
    segments = pd.read_csv(required["segments"])
    anomalies = pd.read_csv(required["anomalies"])
    data = load_analysis_data(get_engine(settings.database_url))
    posts = data.posts.copy()
    settings.exports_dir.mkdir(parents=True, exist_ok=True)
    outputs: list[Path] = []

    dim_topic = _topic_dimension(posts, assignments, topics)
    dim_author = _author_dimension(posts, get_engine(settings.database_url), segments)
    all_dates = pd.concat(
        [
            posts["created_at"],
            data.daily["day"],
            pd.to_datetime(anomalies.get("day", pd.Series(dtype="datetime64[ns, UTC]")), utc=True),
        ],
        ignore_index=True,
    )
    dim_date = _dim_date(all_dates)

    post_frame = posts.merge(assignments[["id", "topic"]], on="id", how="left")
    post_frame["topic_id"] = post_frame["topic"].fillna(-1).astype(int)
    fact_post = pd.DataFrame(
        {
            "post_id": post_frame["id"].astype(int),
            "topic_id": post_frame["topic_id"].astype(int),
            "author_id": post_frame["author_id"],
            "date_key": _date_key(post_frame["created_at"]),
            "score": post_frame["score"].fillna(0).astype(int),
            "view_count": post_frame["view_count"].fillna(0).astype(int),
            "answer_count": post_frame["answer_count"].fillna(0).astype(int),
            "is_answered": post_frame["is_answered"].fillna(False).astype(bool),
            "accepted_answer_id": post_frame["accepted_answer_id"],
            "source_site": post_frame["source_site"],
            "title": post_frame["title"],
            "url": post_frame["url"],
        }
    )

    anomaly_days = set(pd.to_datetime(anomalies["day"], utc=True).dt.normalize())
    daily = data.daily.copy()
    daily["is_anomaly"] = daily["day"].dt.normalize().isin(anomaly_days)
    fact_daily = pd.DataFrame(
        {
            "date_key": _date_key(daily["day"]),
            "posts_created": daily["posts_created"].astype(int),
            "answers_created": daily["answers_created"].astype(int),
            "unique_authors": daily["unique_authors"].astype(int),
            "answered_count": daily["answered_count"].astype(int),
            "total_views": daily["total_views"].astype(int),
            "is_anomaly": daily["is_anomaly"].astype(bool),
        }
    )

    fact_anomaly = pd.DataFrame(
        {
            "date_key": _date_key(anomalies["day"]),
            "metric": anomalies["metric"],
            "value": anomalies["value"],
            "expected": anomalies["expected"],
            "deviation": anomalies["deviation"],
            "methods": anomalies["methods"],
            "direction": anomalies["direction"],
        }
    )

    frames = {
        "dim_topic.csv": dim_topic,
        "dim_author.csv": dim_author,
        "dim_date.csv": dim_date,
        "fact_post.csv": fact_post,
        "fact_engagement_daily.csv": fact_daily,
        "fact_anomaly.csv": fact_anomaly,
    }
    for name, frame in frames.items():
        path = settings.exports_dir / name
        frame.to_csv(path, index=False)
        outputs.append(path)
    measures = settings.exports_dir / "powerbi_measures.md"
    _write_powerbi_measures(measures)
    outputs.append(measures)
    views_path = Path(__file__).resolve().parents[2] / "sql" / "views.sql"
    _write_sql_views(views_path)
    outputs.append(views_path)
    logger.info("Wrote star-schema exports to %s", settings.exports_dir)
    return outputs
