"""Normalize scraped questions and load them into the analytics warehouse."""

from __future__ import annotations

import json
import logging
import time
from dataclasses import replace
from datetime import datetime
from pathlib import Path
from typing import Any

from sqlalchemy import Engine, func, select, text
from sqlalchemy.orm import Session

from ctr.config import Settings
from ctr.db import get_engine, init_db, session_scope, upsert_many
from ctr.models import Author, EngagementDaily, Post, PostTag, Tag
from ctr.scrape import StackExchangeClient, _utc_datetime, parse_question

logger = logging.getLogger(__name__)
SQL_DIR = Path(__file__).resolve().parents[2] / "sql"


def _record_from_sample(raw: dict[str, Any]) -> dict[str, Any]:
    record = dict(raw)
    record["created_at"] = _utc_datetime(record.get("created_at"))
    record.setdefault("author_reputation", 0)
    record.setdefault("author_name", "anonymous")
    record.setdefault("author_id", None)
    record.setdefault("score", 0)
    record.setdefault("view_count", 0)
    record.setdefault("answer_count", 0)
    record.setdefault("is_answered", False)
    record.setdefault("accepted_answer_id", None)
    record.setdefault("source_site", "datascience")
    record.setdefault("tags", [])
    record["id"] = int(record["id"])
    return record


def _normalize_records(
    raw_questions: list[dict[str, Any]], site: str
) -> list[dict[str, Any]]:
    normalized: list[dict[str, Any]] = []
    for raw in raw_questions:
        if "question_id" in raw:
            normalized.append(parse_question(raw, site=site))
        else:
            normalized.append(_record_from_sample(raw))
    return normalized


def _load_sample(path: Path) -> list[dict[str, Any]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, list):
        raise ValueError(f"Sample file must contain a JSON list: {path}")
    return [_record_from_sample(item) for item in payload]


def _count(session: Session, model: type[Any]) -> int:
    value = session.scalar(select(func.count()).select_from(model))
    return int(value or 0)


def _load_engagement_sql() -> str:
    return (SQL_DIR / "engagement_daily.sql").read_text(encoding="utf-8")


def run_ingest(
    settings: Settings,
    *,
    offline: bool = False,
    sample_path: Path | str | None = None,
    engine: Engine | None = None,
    write_stats: bool = True,
) -> dict[str, Any]:
    """Scrape or replay questions, upsert entities, and rebuild daily engagement.

    When ``offline`` is true and raw API pages are unavailable, the committed
    sample is replayed so CI can run the full pipeline without network access.
    """
    started = time.perf_counter()
    active_engine = engine or get_engine(settings.database_url)
    init_db(active_engine)

    raw_questions: list[dict[str, Any]] = []
    source = "api"
    sample = Path(sample_path) if sample_path else settings.sample_dir / "posts_sample.json"
    if sample_path is not None:
        records = _load_sample(Path(sample_path))
        source = "sample"
    elif offline:
        client = StackExchangeClient(
            site=settings.site,
            page_size=settings.page_size,
            delay=settings.request_delay,
            cache_dir=settings.cache_dir,
            raw_dir=settings.raw_dir,
            offline=True,
        )
        try:
            raw_questions = client.load_offline_questions()
            records = _normalize_records(raw_questions, settings.site)
            source = "raw-replay"
        except FileNotFoundError:
            logger.warning("No raw replay data found; falling back to committed sample")
            records = _load_sample(sample)
            source = "sample"
    else:
        client = StackExchangeClient(
            site=settings.site,
            page_size=settings.page_size,
            delay=settings.request_delay,
            cache_dir=settings.cache_dir,
            raw_dir=settings.raw_dir,
        )
        for page in client.fetch_questions(settings.pages):
            raw_questions.extend(page)
        records = _normalize_records(raw_questions, settings.site)

    if not records:
        raise RuntimeError("No questions were ingested; check the scrape configuration")

    posts: list[dict[str, Any]] = []
    author_state: dict[int, dict[str, Any]] = {}
    tag_names: set[str] = set()
    post_tag_names: dict[int, list[str]] = {}
    for record in records:
        post_row = {
            key: record[key]
            for key in (
                "id",
                "title",
                "body_text",
                "author_id",
                "author_name",
                "created_at",
                "score",
                "view_count",
                "answer_count",
                "is_answered",
                "accepted_answer_id",
                "url",
                "source_site",
            )
        }
        posts.append(post_row)
        author_id = post_row["author_id"]
        if author_id is not None:
            observed = author_state.get(author_id)
            if observed is None:
                author_state[author_id] = {
                    "id": author_id,
                    "name": post_row["author_name"] or "anonymous",
                    "reputation": int(record.get("author_reputation", 0)),
                    "first_seen_at": post_row["created_at"],
                    "last_seen_at": post_row["created_at"],
                }
            else:
                observed["first_seen_at"] = min(
                    observed["first_seen_at"], post_row["created_at"]
                )
                observed["last_seen_at"] = max(observed["last_seen_at"], post_row["created_at"])
                observed["reputation"] = max(
                    observed["reputation"], int(record.get("author_reputation", 0))
                )
        names = [name for name in record.get("tags", []) if name]
        post_tag_names[int(post_row["id"])] = names
        tag_names.update(names)

    with session_scope(active_engine) as session:
        post_count = upsert_many(session, Post, posts, ["id"])
        author_count = upsert_many(
            session, Author, list(author_state.values()), ["id"]
        )
        if tag_names:
            upsert_many(
                session,
                Tag,
                [{"name": name} for name in sorted(tag_names)],
                ["name"],
            )
        tag_map = {
            name: tag_id
            for tag_id, name in session.execute(select(Tag.id, Tag.name)).all()
        }
        links = [
            {"post_id": post_id, "tag_id": tag_map[name]}
            for post_id, names in post_tag_names.items()
            for name in names
            if name in tag_map
        ]
        upsert_many(session, PostTag, links, ["post_id", "tag_id"])

        session.execute(text("DELETE FROM engagement_daily"))
        daily_sql = _load_engagement_sql()
        session.execute(text(daily_sql))
        session.flush()

        counts = {
            "posts": _count(session, Post),
            "authors": _count(session, Author),
            "tags": _count(session, Tag),
            "post_tags": _count(session, PostTag),
            "engagement_daily": _count(session, EngagementDaily),
        }
        date_min, date_max = session.execute(
            select(func.min(Post.created_at), func.max(Post.created_at))
        ).one()

    manifest_path = settings.raw_dir / settings.site / "manifest.json"
    raw_page_count = 0
    if manifest_path.exists():
        try:
            raw_page_count = len(json.loads(manifest_path.read_text(encoding="utf-8")))
        except json.JSONDecodeError:
            logger.warning("Could not read raw manifest at %s", manifest_path)
    elapsed = round(time.perf_counter() - started, 2)
    stats: dict[str, Any] = {
        "generated_at": datetime.now().astimezone().isoformat(),
        "source": source,
        "source_site": settings.site,
        "offline": offline,
        "raw_pages": raw_page_count,
        **counts,
        "date_min": date_min.isoformat() if date_min else None,
        "date_max": date_max.isoformat() if date_max else None,
        "rows_upserted": {
            "posts": post_count,
            "authors": author_count,
            "tags": len(tag_names),
            "post_tags": len(links),
        },
        "run_seconds": elapsed,
    }
    if write_stats:
        settings.reports_dir.mkdir(parents=True, exist_ok=True)
        output = settings.reports_dir / "ingest_stats.json"
        output.write_text(json.dumps(stats, indent=2), encoding="utf-8")
        logger.info("Wrote ingest statistics to %s", output)

    for table, count in counts.items():
        logger.info("Table %-18s %s rows", table, f"{count:,}")
    logger.info("Ingest finished in %.2f seconds", elapsed)
    return stats


def run_offline_sample(
    settings: Settings,
    *,
    sample_path: Path | str | None = None,
    engine: Engine | None = None,
    write_stats: bool = False,
) -> dict[str, Any]:
    """Convenience wrapper used by tests and the CI smoke job."""
    path = Path(sample_path) if sample_path else settings.sample_dir / "posts_sample.json"
    sampled = replace(settings, pages=1)
    return run_ingest(
        sampled,
        offline=True,
        sample_path=path,
        engine=engine,
        write_stats=write_stats,
    )
