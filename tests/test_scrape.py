"""Tests for the Stack Exchange API client and HTML fallback parser."""

from __future__ import annotations

import json
from datetime import UTC
from unittest.mock import Mock

import pytest
from tests.conftest import FIXTURES

import ctr.scrape
from ctr.scrape import (
    StackExchangeClient,
    fetch_for_settings,
    parse_answer,
    parse_question,
    scrape_html_fallback,
)


def test_parse_question_fixture() -> None:
    raw = json.loads((FIXTURES / "question_api.json").read_text(encoding="utf-8"))
    parsed = parse_question(raw, site="stats")
    assert parsed["id"] == raw["question_id"]
    assert "<strong>" not in parsed["body_text"]
    assert "confidence interval" in parsed["body_text"]
    assert parsed["created_at"].tzinfo is not None
    assert parsed["created_at"].astimezone(UTC)
    assert parsed["author_name"] == raw["owner"]["display_name"]
    assert parsed["source_site"] == "stats"
    assert parsed["tags"] == raw["tags"]


def test_parse_question_generates_anonymous_author() -> None:
    parsed = parse_question(
        {
            "question_id": 5,
            "title": "No owner",
            "creation_date": 1_700_000_000,
            "tags": [],
        },
        site="stats",
    )
    assert parsed["author_id"] is not None
    assert parsed["author_id"] < 0
    assert parsed["author_name"] == "anonymous"


def test_parse_answer() -> None:
    parsed = parse_answer(
        {
            "answer_id": 11,
            "question_id": 10,
            "creation_date": 1_700_000_000,
            "is_accepted": True,
            "score": 2,
            "owner": {"user_id": 3, "display_name": "A", "reputation": 9},
        },
        site="stats",
    )
    assert parsed["question_id"] == 10
    assert parsed["is_accepted"] is True
    assert parsed["author_reputation"] == 9


def test_html_fallback_fixture() -> None:
    html = (FIXTURES / "questions.html").read_text(encoding="utf-8")
    posts = scrape_html_fallback(
        "https://stats.stackexchange.com/questions", html=html, site="stats"
    )
    assert len(posts) == 3
    first = posts[0]
    assert first["id"] == 101
    assert first["title"] == "How do I interpret a p-value?"
    assert first["answer_count"] == 3
    assert first["is_answered"] is True
    assert first["tags"] == ["hypothesis-testing", "p-value"]
    assert first["created_at"].year == 2025


def test_html_fallback_discourse_and_network_path() -> None:
    html = """
    <table><tr class="topic-list-item">
      <td class="posts">4</td><td class="views">99</td>
      <td class="main-link"><a class="title" href="/t/example-topic/321">Example</a></td>
      <td class="posters"><a>Dana</a></td>
      <td><a class="discourse-tag">forecasting</a></td>
      <td><time datetime="2025-02-01T10:00:00Z"></time></td>
    </tr></table>
    """
    response = Mock()
    response.text = html
    response.raise_for_status = Mock()
    session = Mock()
    session.get.return_value = response
    posts = scrape_html_fallback(
        "https://forum.example.org/latest", session=session
    )
    assert posts[0]["id"] == 321
    assert posts[0]["source_site"] == "forum"


def _response(status: int, payload: dict | None = None) -> Mock:
    response = Mock()
    response.status_code = status
    response.raise_for_status = Mock()
    response.json.return_value = payload or {"items": [], "has_more": False}
    return response


def test_retry_on_429_then_success(tmp_path) -> None:
    session = Mock()
    session.get.side_effect = [_response(429), _response(200, {"items": [1], "has_more": False})]
    client = StackExchangeClient(
        site="stats",
        delay=0,
        cache_dir=tmp_path / "cache",
        raw_dir=tmp_path / "raw",
        session=session,
    )
    client._sleep = lambda seconds: None  # type: ignore[method-assign]
    payload = client._request("https://api.example.test/questions", {"site": "stats"})
    assert payload["items"] == [1]
    assert session.get.call_count == 2


def test_cache_hit_avoids_second_request(tmp_path) -> None:
    session = Mock()
    session.get.return_value = _response(200, {"items": [1], "has_more": False})
    client = StackExchangeClient(
        site="stats",
        delay=0,
        cache_dir=tmp_path / "cache",
        raw_dir=tmp_path / "raw",
        session=session,
    )
    client._sleep = lambda seconds: None  # type: ignore[method-assign]
    first = client._request("https://api.example.test/questions", {"site": "stats"})
    second_session = Mock()
    second = StackExchangeClient(
        site="stats",
        delay=0,
        cache_dir=tmp_path / "cache",
        raw_dir=tmp_path / "raw",
        session=second_session,
    )
    second._sleep = lambda seconds: None  # type: ignore[method-assign]
    cached = second._request("https://api.example.test/questions", {"site": "stats"})
    assert first == cached
    second_session.get.assert_not_called()
    assert (tmp_path / "raw" / "stats" / "manifest.json").exists()


def test_offline_request_requires_raw_payload(tmp_path) -> None:
    client = StackExchangeClient(
        site="stats",
        delay=0,
        cache_dir=tmp_path / "cache",
        raw_dir=tmp_path / "raw",
        offline=True,
    )
    with pytest.raises(FileNotFoundError):
        client._request("https://api.example.test/questions", {"site": "stats"})


def test_fetch_questions_and_answers_use_pages(tmp_path) -> None:
    client = StackExchangeClient(
        site="stats", delay=0, cache_dir=tmp_path / "cache", raw_dir=tmp_path / "raw"
    )
    pages = [
        {"items": [{"question_id": 1, "creation_date": 10}], "has_more": True},
        {"items": [{"question_id": 2, "creation_date": 9}], "has_more": False},
    ]
    client._request = lambda base, params: pages.pop(0)  # type: ignore[method-assign]
    questions = list(client.fetch_questions(10))
    assert [item["question_id"] for item in questions[0]] == [1]
    assert len(questions) == 2

    answer_pages = [
        {"items": [{"answer_id": 1, "creation_date": 10}], "has_more": False}
    ]
    client._request = lambda base, params: answer_pages.pop(0)  # type: ignore[method-assign]
    answers = list(client.fetch_answers(fromdate=1))
    assert answers[0][0]["answer_id"] == 1


def test_fetch_for_settings_offline(tmp_path, settings, monkeypatch) -> None:
    class DummyClient:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

        def load_offline_questions(self):
            return [{"question_id": 1}]

    monkeypatch.setattr(ctr.scrape, "StackExchangeClient", DummyClient)
    from dataclasses import replace

    replayed = fetch_for_settings(replace(settings, site="stats"), offline=True)
    assert replayed[0]["question_id"] == 1


def test_parse_question_rejects_missing_id() -> None:
    with pytest.raises(ValueError):
        parse_question({"title": "bad"})
