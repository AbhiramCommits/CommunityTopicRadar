"""Stack Exchange API client and HTML fallback scraper."""

from __future__ import annotations

import hashlib
import json
import logging
import re
import time
from collections.abc import Iterator, Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from urllib.parse import urlencode, urljoin, urlparse

import requests
from bs4 import BeautifulSoup

from ctr.config import Settings

logger = logging.getLogger(__name__)

API_ROOT = "https://api.stackexchange.com/2.3"
RETRYABLE_STATUS = {429, 500, 502, 503, 504}
MAX_API_PAGE = 25


def _clean_html(value: str | None) -> str:
    if not value:
        return ""
    if "<" not in value:
        return " ".join(value.split())
    text = BeautifulSoup(value, "lxml").get_text(" ")
    return " ".join(text.split())


def _utc_datetime(timestamp: int | float | str | None) -> datetime:
    if timestamp is None:
        return datetime.now(UTC)
    if isinstance(timestamp, str):
        parsed = datetime.fromisoformat(timestamp.replace("Z", "+00:00"))
        if parsed.tzinfo is None:
            parsed = parsed.replace(tzinfo=UTC)
        return parsed.astimezone(UTC)
    return datetime.fromtimestamp(float(timestamp), tz=UTC)


def _synthetic_author_id(name: str | None) -> int | None:
    if not name:
        return None
    digest = hashlib.blake2b(name.encode("utf-8"), digest_size=4).hexdigest()
    return -(int(digest, 16) % 2_000_000_000) - 1


def _site_from_url(url: str) -> str:
    host = urlparse(url).netloc.lower()
    if host.startswith("www."):
        host = host[4:]
    if host.endswith(".stackexchange.com"):
        return host.removesuffix(".stackexchange.com")
    return host.split(".")[0] or "html"


def parse_question(raw: dict[str, Any], site: str | None = None) -> dict[str, Any]:
    """Normalize a Stack Exchange question payload into warehouse fields."""
    owner = raw.get("owner") or {}
    author_name = owner.get("display_name") or "anonymous"
    author_id = owner.get("user_id") or _synthetic_author_id(author_name)
    source_site = site or raw.get("source_site") or "datascience"
    question_id = raw.get("question_id", raw.get("id"))
    if question_id is None:
        raise ValueError("Question payload has no question_id")
    link = raw.get("link") or (
        f"https://{source_site}.stackexchange.com/questions/{int(question_id)}"
    )
    return {
        "id": int(question_id),
        "title": _clean_html(raw.get("title", "")),
        "body_text": _clean_html(raw.get("body")),
        "author_id": author_id,
        "author_name": author_name,
        "author_reputation": int(owner.get("reputation") or 0),
        "created_at": _utc_datetime(raw.get("creation_date")),
        "score": int(raw.get("score") or 0),
        "view_count": int(raw.get("view_count") or 0),
        "answer_count": int(raw.get("answer_count") or 0),
        "is_answered": bool(raw.get("is_answered", False)),
        "accepted_answer_id": raw.get("accepted_answer_id"),
        "url": link,
        "source_site": source_site,
        "tags": [str(tag) for tag in raw.get("tags", [])],
    }


def _first_number(container: BeautifulSoup, selectors: Sequence[str]) -> int:
    for selector in selectors:
        element = container.select_one(selector)
        if element is None:
            continue
        match = re.search(r"[\d,]+", element.get_text(" ", strip=True))
        if match:
            return int(match.group(0).replace(",", ""))
    return 0


def _first_text(container: BeautifulSoup, selectors: Sequence[str]) -> str:
    for selector in selectors:
        element = container.select_one(selector)
        if element is not None:
            text = element.get_text(" ", strip=True)
            if text:
                return text
    return ""


def _parse_html_date(container: BeautifulSoup) -> datetime:
    for selector in ("time[datetime]", "[data-created-at]", ".relativetime[title]"):
        element = container.select_one(selector)
        if element is None:
            continue
        value = element.get("datetime") or element.get("data-created-at") or element.get("title")
        if not value:
            continue
        for pattern in ("%Y-%m-%dT%H:%M:%S%z", "%Y-%m-%dT%H:%M:%S", "%Y-%m-%d %H:%M:%S"):
            try:
                parsed = datetime.strptime(str(value).replace("Z", "+0000"), pattern)
                if parsed.tzinfo is None:
                    parsed = parsed.replace(tzinfo=UTC)
                return parsed.astimezone(UTC)
            except ValueError:
                continue
        try:
            return _utc_datetime(str(value))
        except ValueError:
            logger.debug("Could not parse HTML timestamp %r", value)
    return datetime.now(UTC)


def _html_row_to_post(container: BeautifulSoup, url: str, site: str) -> dict[str, Any] | None:
    title_link = container.select_one(
        "h3 a, a.s-post-summary--content-title, "
        "a.question-hyperlink, a.title, a[href*='/questions/']"
    )
    if title_link is None:
        return None
    href = title_link.get("href", "")
    absolute = urljoin(url, href)
    identifier_match = re.search(r"/(?:questions|t)/(?:[^/]+/)?(\d+)", absolute)
    if identifier_match is None:
        identifier_match = re.search(r"(\d+)(?:/)?$", absolute)
    question_id = (
        int(identifier_match.group(1)) if identifier_match else _synthetic_author_id(absolute)
    )
    if question_id is None:
        return None
    author_name = _first_text(
        container,
        [
            ".s-user-card--link",
            ".user-details a",
            ".posters a",
            ".user a",
        ],
    )
    author_id = _synthetic_author_id(author_name) if author_name else None
    tags = [
        tag.get_text(" ", strip=True)
        for tag in container.select(".post-tag, .s-post-summary--meta-tag, .discourse-tag")
    ]
    title = title_link.get_text(" ", strip=True)
    excerpt = _first_text(
        container,
        [
            ".s-post-summary--content-excerpt",
            ".topic-excerpt",
            ".excerpt",
            ".question-summary .excerpt",
        ],
    )
    score = _first_number(
        container,
        [
            ".s-post-summary--stats-item:first-child .s-post-summary--stats-item-number",
            ".vote-count-post",
            ".posts",
            "[data-score]",
        ],
    )
    answer_count = _first_number(
        container,
        [
            ".s-post-summary--stats-item:nth-child(2) .s-post-summary--stats-item-number",
            ".status strong",
            ".answers",
        ],
    )
    view_count = _first_number(
        container,
        [
            ".s-post-summary--stats-item:nth-child(3) .s-post-summary--stats-item-number",
            ".views",
            ".view-count",
        ],
    )
    classes = " ".join(container.get("class", []))
    is_answered = answer_count > 0 or "answered" in classes or "has-accepted-answer" in classes
    return {
        "id": abs(int(question_id)),
        "title": title,
        "body_text": excerpt,
        "author_id": author_id,
        "author_name": author_name or "anonymous",
        "author_reputation": 0,
        "created_at": _parse_html_date(container),
        "score": score,
        "view_count": view_count,
        "answer_count": answer_count,
        "is_answered": is_answered,
        "accepted_answer_id": None,
        "url": absolute,
        "source_site": site,
        "tags": tags,
    }


def scrape_html_fallback(
    url: str,
    *,
    html: str | None = None,
    session: requests.Session | None = None,
    site: str | None = None,
) -> list[dict[str, Any]]:
    """Parse a Stack Exchange list page or a Discourse topic list.

    Supplying ``html`` is the pure, network-free path used by tests. Without it,
    the page is fetched with a short request timeout.
    """
    if html is None:
        active_session = session or requests.Session()
        response = active_session.get(
            url, timeout=30, headers={"User-Agent": "CommunityTopicRadar/0.1"}
        )
        response.raise_for_status()
        html = response.text
    soup = BeautifulSoup(html, "lxml")
    containers = soup.select(
        "div.s-post-summary, div.question-summary, tr.topic-list-item, div.topic-list-item"
    )
    source_site = site or _site_from_url(url)
    posts: list[dict[str, Any]] = []
    for container in containers:
        parsed = _html_row_to_post(container, url, source_site)
        if parsed is not None:
            posts.append(parsed)
    return posts


class StackExchangeClient:
    """Polite, cached Stack Exchange API client with offline replay support."""

    def __init__(
        self,
        site: str = "datascience",
        page_size: int = 100,
        delay: float = 1.0,
        cache_dir: Path | str = Path("data/cache"),
        raw_dir: Path | str = Path("data/raw"),
        session: requests.Session | None = None,
        offline: bool = False,
        timeout: float = 30.0,
        max_tries: int = 4,
    ) -> None:
        self.site = site
        self.page_size = page_size
        self.delay = delay
        self.cache_dir = Path(cache_dir)
        self.raw_dir = Path(raw_dir)
        self.session = session or requests.Session()
        self.offline = offline
        self.timeout = timeout
        self.max_tries = max_tries
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.raw_dir.mkdir(parents=True, exist_ok=True)

    @staticmethod
    def _hash_url(url: str) -> str:
        return hashlib.sha256(url.encode("utf-8")).hexdigest()[:20]

    @property
    def _manifest_path(self) -> Path:
        return self.raw_dir / self.site / "manifest.json"

    def _cache_path(self, url: str) -> Path:
        return self.cache_dir / f"{self._hash_url(url)}.json"

    def _raw_path(self, url: str) -> Path:
        return self.raw_dir / self.site / f"{self._hash_url(url)}.json"

    def _record_raw(self, url: str, payload: dict[str, Any]) -> None:
        path = self._raw_path(url)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        manifest: list[dict[str, Any]] = []
        if self._manifest_path.exists():
            try:
                manifest = json.loads(self._manifest_path.read_text(encoding="utf-8"))
            except json.JSONDecodeError:
                logger.warning("Ignoring malformed raw manifest at %s", self._manifest_path)
        if not any(entry.get("file") == path.name for entry in manifest):
            manifest.append(
                {
                    "file": path.name,
                    "url": url,
                    "fetched_at": datetime.now(UTC).isoformat(),
                }
            )
            self._manifest_path.write_text(
                json.dumps(manifest, indent=2), encoding="utf-8"
            )

    def _sleep(self, seconds: float) -> None:
        if seconds > 0:
            time.sleep(seconds)

    def _request(self, base_url: str, params: dict[str, Any]) -> dict[str, Any]:
        query = urlencode(sorted(params.items()))
        url = f"{base_url}?{query}"
        cache_path = self._cache_path(url)
        raw_path = self._raw_path(url)

        if self.offline:
            if not raw_path.exists():
                raise FileNotFoundError(f"No raw payload for offline request: {url}")
            return json.loads(raw_path.read_text(encoding="utf-8"))

        if cache_path.exists():
            logger.debug("Cache hit for %s", url)
            payload = json.loads(cache_path.read_text(encoding="utf-8"))
            if not raw_path.exists():
                self._record_raw(url, payload)
            return payload

        last_error: Exception | None = None
        for attempt in range(self.max_tries):
            self._sleep(self.delay)
            try:
                response = self.session.get(
                    base_url,
                    params=params,
                    timeout=self.timeout,
                    headers={"User-Agent": "CommunityTopicRadar/0.1"},
                )
            except requests.RequestException as exc:
                last_error = exc
                if attempt == self.max_tries - 1:
                    break
                self._sleep(2**attempt)
                continue

            if response.status_code in RETRYABLE_STATUS:
                if attempt == self.max_tries - 1:
                    response.raise_for_status()
                self._sleep(2**attempt)
                continue
            response.raise_for_status()
            payload = response.json()
            if "error_message" in payload:
                raise RuntimeError(f"Stack Exchange API error: {payload['error_message']}")
            cache_path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
            self._record_raw(url, payload)
            backoff = payload.get("backoff")
            if backoff:
                self._sleep(float(backoff))
            return payload
        raise RuntimeError(
            f"API request failed after {self.max_tries} attempts: {url}"
        ) from last_error

    def fetch_questions(self, pages: int) -> Iterator[list[dict[str, Any]]]:
        """Yield pages of raw questions, continuing past the 25-page API cap."""
        fetched = 0
        page = 1
        todate: int | None = None
        base_url = f"{API_ROOT}/questions"
        while fetched < pages:
            params: dict[str, Any] = {
                "order": "desc",
                "sort": "creation",
                "site": self.site,
                "filter": "withbody",
                "pagesize": self.page_size,
                "page": page,
            }
            if todate is not None:
                params["todate"] = todate
            payload = self._request(base_url, params)
            items = payload.get("items", [])
            if not items:
                break
            yield items
            fetched += 1
            timestamps = [int(item["creation_date"]) for item in items if "creation_date" in item]
            if not payload.get("has_more", False):
                break
            if page >= MAX_API_PAGE:
                if not timestamps:
                    break
                todate = min(timestamps) - 1
                page = 1
            else:
                page += 1
        logger.info("Fetched %d raw question pages from %s", fetched, self.site)

    def load_offline_questions(self) -> list[dict[str, Any]]:
        """Replay every raw API page recorded for this site."""
        if not self._manifest_path.exists():
            raise FileNotFoundError(f"No raw manifest at {self._manifest_path}")
        manifest = json.loads(self._manifest_path.read_text(encoding="utf-8"))
        raw: list[dict[str, Any]] = []
        for entry in manifest:
            path = self._manifest_path.parent / entry["file"]
            payload = json.loads(path.read_text(encoding="utf-8"))
            raw.extend(payload.get("items", []))
        logger.info("Replayed %d raw questions from %s", len(raw), self._manifest_path)
        return raw


def fetch_for_settings(settings: Settings, offline: bool = False) -> list[dict[str, Any]]:
    """Fetch or replay raw question payloads using settings."""
    client = StackExchangeClient(
        site=settings.site,
        page_size=settings.page_size,
        delay=settings.request_delay,
        cache_dir=settings.cache_dir,
        raw_dir=settings.raw_dir,
        offline=offline,
    )
    if offline:
        return client.load_offline_questions()
    raw: list[dict[str, Any]] = []
    for page in client.fetch_questions(settings.pages):
        raw.extend(page)
    return raw
