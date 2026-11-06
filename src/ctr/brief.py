"""Plain-language topic briefs with an offline deterministic fallback."""

from __future__ import annotations

import hashlib
import json
import logging
from pathlib import Path
from typing import Any

import pandas as pd
from openai import OpenAI

from ctr.analysis.pipeline import load_analysis_data
from ctr.config import Settings, load_settings
from ctr.db import get_engine

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = (
    "You write 3-sentence briefs for a non-technical community program manager. "
    "No jargon. State what the topic is, why it matters, and the recommended action."
)


class TemplateSummarizer:
    """Deterministic, offline fallback summarizer."""

    def _action(self, topic: dict[str, Any]) -> str:
        gap = float(topic.get("gap_score", 0.0))
        answer_rate = float(topic.get("answer_rate", 0.0))
        if gap >= 0.3 or answer_rate < 0.45:
            return (
                "commission a short plain-language explainer and ask two experienced "
                "volunteers to watch new questions on it"
            )
        if answer_rate < 0.7:
            return "route new questions on it to the most experienced volunteers"
        return "keep monitoring it and refresh the explainer every quarter"

    def summarize(self, topic: dict[str, Any]) -> str:
        """Return a deterministic three-sentence brief from topic facts."""
        label = str(topic["label"])
        posts = int(topic.get("posts", 0))
        answer_rate = float(topic.get("answer_rate", 0.0))
        views = int(topic.get("median_views", 0))
        trend = float(topic.get("pct_change", 0.0))
        if trend > 0:
            direction = "grew"
        elif trend < 0:
            direction = "fell"
        else:
            direction = "held steady"
        return (
            f"{label} is a recurring community topic with {posts} posts and "
            f"{answer_rate:.0%} of them answered. "
            f"Question volume {direction} week over week, and the typical post "
            f"receives {views} views. "
            f"Program owners should {self._action(topic)}."
        )


def _cache_path(settings: Settings) -> Path:
    return settings.cache_dir / "briefs.json"


def _cache_key(topic: dict[str, Any]) -> str:
    fingerprint = {
        "label": topic.get("label"),
        "terms": topic.get("terms"),
        "posts": topic.get("posts"),
        "current_count": topic.get("current_count"),
        "prior_count": topic.get("prior_count"),
        "answer_rate": topic.get("answer_rate"),
        "gap_score": topic.get("gap_score"),
    }
    return hashlib.sha256(
        json.dumps(fingerprint, sort_keys=True, default=str).encode("utf-8")
    ).hexdigest()[:24]


def _load_cache(path: Path) -> dict[str, str]:
    if not path.exists():
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
        return payload if isinstance(payload, dict) else {}
    except json.JSONDecodeError:
        logger.warning("Ignoring malformed brief cache at %s", path)
        return {}


def _save_cache(path: Path, cache: dict[str, str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(cache, indent=2, sort_keys=True), encoding="utf-8")


def build_prompt(topic: dict[str, Any]) -> str:
    """Assemble the compact factual prompt sent to the language model."""
    examples = topic.get("examples", [])
    example_lines = "\n".join(f"- {title}" for title in examples[:3]) or "- none available"
    return (
        f"Topic label: {topic['label']}\n"
        f"Top terms: {', '.join(topic.get('terms', [])[:10])}\n"
        f"Posts: {topic.get('posts', 0)}\n"
        f"Current-week posts: {topic.get('current_count', 0)}\n"
        f"Prior-week posts: {topic.get('prior_count', 0)}\n"
        f"Week-over-week change: {float(topic.get('pct_change', 0.0)):+.1f}%\n"
        f"Answer rate: {float(topic.get('answer_rate', 0.0)):.1%}\n"
        f"Gap score: {float(topic.get('gap_score', 0.0)):.3f}\n"
        f"Representative post titles:\n{example_lines}"
    )


def summarize_topic(
    topic: dict[str, Any],
    client: Any | None = None,
    settings: Settings | None = None,
) -> str:
    """Summarize a topic with OpenAI, falling back to a deterministic template.

    Responses (from either path) are cached in data/cache/briefs.json under a
    hash of the topic facts.
    """
    active_settings = settings or load_settings()
    path = _cache_path(active_settings)
    cache = _load_cache(path)
    key = _cache_key(topic)
    if key in cache:
        return cache[key]

    active_client = client
    if active_client is None and active_settings.openai_api_key:
        active_client = OpenAI(api_key=active_settings.openai_api_key)

    summary: str | None = None
    if active_client is not None:
        for attempt in range(2):
            try:
                response = active_client.chat.completions.create(
                    model=active_settings.openai_model,
                    messages=[
                        {"role": "system", "content": SYSTEM_PROMPT},
                        {"role": "user", "content": build_prompt(topic)},
                    ],
                    max_tokens=180,
                    temperature=0.3,
                )
                candidate = response.choices[0].message.content
                if candidate and candidate.strip():
                    summary = candidate.strip()
                    break
            except Exception:
                logger.warning("OpenAI brief attempt %d failed", attempt + 1, exc_info=True)
    if summary is None:
        summary = TemplateSummarizer().summarize(topic)
        logger.info("Using deterministic template brief for topic %s", topic.get("label"))

    cache[key] = summary
    _save_cache(path, cache)
    return summary


def _topic_prompt_data(settings: Settings) -> list[dict[str, Any]]:
    reports = settings.reports_dir
    topics_path = reports / "topics.csv"
    assignments_path = reports / "topic_assignments.csv"
    if not topics_path.exists() or not assignments_path.exists():
        raise FileNotFoundError("Run `python -m ctr.cli analyze` before generating briefs")
    topics = pd.read_csv(topics_path)
    assignments = pd.read_csv(assignments_path)
    data = load_analysis_data(get_engine(settings.database_url))
    posts = data.posts.merge(assignments[["id", "topic"]], on="id", how="left")
    posts = posts.dropna(subset=["topic"])
    posts["topic"] = posts["topic"].astype(int)
    terms_by_topic = {
        int(topic): group.sort_values("rank")["term"].astype(str).tolist()
        for topic, group in topics.groupby("topic")
    }
    newest = posts["created_at"].max()
    current_end = newest.normalize() - pd.Timedelta(days=1)
    current_start = current_end - pd.Timedelta(days=6)
    prior_end = current_start - pd.Timedelta(days=1)
    prior_start = prior_end - pd.Timedelta(days=6)

    stats = posts.groupby("topic").agg(
        posts=("id", "count"),
        answer_rate=("is_answered", "mean"),
        median_views=("view_count", "median"),
        accepted_rate=("accepted_answer_id", lambda values: float(values.notna().mean())),
    )
    demand = (
        stats["posts"].rank(pct=True, method="average")
        + stats["median_views"].rank(pct=True, method="average")
    ) / 2.0
    stats["gap_score"] = demand * (1.0 - stats["answer_rate"])

    records: list[dict[str, Any]] = []
    for topic_id, row in stats.iterrows():
        subset = posts[posts["topic"] == topic_id]
        current_count = int(
            ((subset["created_at"] >= current_start) & (subset["created_at"] <= current_end)).sum()
        )
        prior_count = int(
            ((subset["created_at"] >= prior_start) & (subset["created_at"] <= prior_end)).sum()
        )
        pct_change = (
            (current_count - prior_count) / prior_count * 100.0
            if prior_count > 0
            else (100.0 if current_count > 0 else 0.0)
        )
        representative = subset.sort_values(
            ["is_answered", "view_count", "score"], ascending=[True, False, False]
        ).head(3)
        terms = terms_by_topic.get(int(topic_id), [])
        label = " / ".join(terms[:3]) or f"Topic {int(topic_id) + 1}"
        records.append(
            {
                "topic_id": int(topic_id),
                "label": label,
                "terms": terms,
                "posts": int(row["posts"]),
                "current_count": current_count,
                "prior_count": prior_count,
                "delta": current_count - prior_count,
                "pct_change": round(pct_change, 2),
                "answer_rate": float(row["answer_rate"]),
                "median_views": float(row["median_views"]),
                "gap_score": float(row["gap_score"]),
                "examples": representative["title"].astype(str).tolist(),
                "urls": representative["url"].astype(str).tolist(),
            }
        )
    records.sort(key=lambda record: (-record["posts"], record["topic_id"]))
    return records


def _action_for(topic: dict[str, Any]) -> str:
    action = TemplateSummarizer()._action(topic)
    return action[0].upper() + action[1:] + "."


def run_briefs(settings: Settings) -> Path:
    """Write reports/topic_briefs.md for every NMF topic cluster."""
    topics = _topic_prompt_data(settings)
    if settings.openai_api_key:
        mode = f"OpenAI model `{settings.openai_model}`"
    else:
        mode = "offline deterministic template fallback (no OPENAI_API_KEY)"
    settings.reports_dir.mkdir(parents=True, exist_ok=True)
    output = settings.reports_dir / "topic_briefs.md"
    lines = [
        "# Topic briefs",
        "",
        f"Generated with: {mode}.",
        "",
    ]
    for topic in topics:
        summary = summarize_topic(topic, settings=settings)
        lines.extend(
            [
                f"## {topic['label']}",
                "",
                summary,
                "",
                "**Numbers behind it**",
                "",
                "| metric | value |",
                "|---|---|",
                f"| posts | {topic['posts']} |",
                f"| latest week posts | {topic['current_count']} |",
                f"| prior week posts | {topic['prior_count']} |",
                f"| week-over-week change | {topic['pct_change']:+.1f}% |",
                f"| answer rate | {topic['answer_rate']:.1%} |",
                f"| median views | {topic['median_views']:.0f} |",
                f"| gap score | {topic['gap_score']:.3f} |",
                "",
                f"**Suggested action for program owners:** {_action_for(topic)}",
                "",
                "**Representative posts**",
                "",
            ]
        )
        for title, url in zip(topic["examples"], topic["urls"], strict=False):
            lines.append(f"- [{title}]({url})")
        lines.append("")
    output.write_text("\n".join(lines), encoding="utf-8")
    logger.info("Wrote %d topic briefs to %s", len(topics), output)
    return output
