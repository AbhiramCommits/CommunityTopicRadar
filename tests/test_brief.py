"""Tests for topic brief generation and caching."""

from __future__ import annotations

import json
from types import SimpleNamespace

from ctr import brief

TOPIC = {
    "label": "lasso / regularization",
    "terms": ["lasso", "regularization", "penalty"],
    "posts": 50,
    "current_count": 10,
    "prior_count": 5,
    "pct_change": 100.0,
    "answer_rate": 0.4,
    "median_views": 170,
    "gap_score": 0.4,
    "examples": ["A", "B", "C"],
    "urls": ["u1", "u2", "u3"],
}


def test_template_summarizer_is_deterministic_and_short() -> None:
    summarizer = brief.TemplateSummarizer()
    first = summarizer.summarize(TOPIC)
    second = summarizer.summarize(TOPIC)
    assert first == second
    assert first.count(".") <= 3
    assert "lasso" in first
    assert "Program owners" in first


def test_build_prompt_contains_facts() -> None:
    prompt = brief.build_prompt(TOPIC)
    assert "lasso / regularization" in prompt
    assert "Week-over-week change: +100.0%" in prompt
    assert "- A" in prompt


def _mock_client(text: str, calls: list[int]) -> SimpleNamespace:
    def create(**kwargs):
        calls.append(1)
        return SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(content=text))]
        )

    return SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create)))


def test_summarize_topic_uses_client_and_caches(settings) -> None:
    calls: list[int] = []
    client = _mock_client("A model-written brief. It is short. It is cached.", calls)
    first = brief.summarize_topic(TOPIC, client=client, settings=settings)
    second = brief.summarize_topic(TOPIC, client=client, settings=settings)
    assert first == second
    assert len(calls) == 1
    cache = json.loads((settings.cache_dir / "briefs.json").read_text())
    assert len(cache) == 1


def test_summarize_topic_retries_then_falls_back(settings) -> None:
    calls: list[int] = []

    def failing_create(**kwargs):
        calls.append(1)
        raise RuntimeError("temporary")

    client = SimpleNamespace(
        chat=SimpleNamespace(
            completions=SimpleNamespace(create=failing_create)
        )
    )
    text = brief.summarize_topic(TOPIC, client=client, settings=settings)
    assert len(calls) == 2
    assert "Program owners" in text


def test_run_briefs_writes_markdown(settings, monkeypatch) -> None:
    monkeypatch.setattr(brief, "_topic_prompt_data", lambda active: [TOPIC])
    output = brief.run_briefs(settings)
    content = output.read_text(encoding="utf-8")
    assert "# Topic briefs" in content
    assert "lasso / regularization" in content
    assert "Suggested action for program owners" in content
    assert "Representative posts" in content
