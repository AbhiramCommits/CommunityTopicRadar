"""Application settings loaded from environment variables."""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv

logger = logging.getLogger(__name__)

DEFAULT_DATABASE_URL = "sqlite:///data/community.db"


@dataclass(frozen=True, slots=True)
class Settings:
    """Runtime configuration for the CommunityTopicRadar pipeline."""

    site: str = "datascience"
    pages: int = 12
    page_size: int = 100
    database_url: str = DEFAULT_DATABASE_URL
    openai_model: str = "gpt-4o-mini"
    openai_api_key: str | None = None
    request_delay: float = 1.0
    data_dir: Path = Path("data")
    reports_dir: Path = Path("reports")
    exports_dir: Path = Path("exports")

    @property
    def raw_dir(self) -> Path:
        """Directory holding raw API and HTML payloads."""
        return self.data_dir / "raw"

    @property
    def cache_dir(self) -> Path:
        """Directory holding the HTTP response cache and brief cache."""
        return self.data_dir / "cache"

    @property
    def sample_dir(self) -> Path:
        """Directory holding the committed offline sample."""
        return self.data_dir / "sample"


def _positive_int(value: str | None, default: int) -> int:
    if value is None or value.strip() == "":
        return default
    parsed = int(value)
    if parsed < 1:
        raise ValueError("Configuration values must be positive integers")
    return parsed


def load_settings() -> Settings:
    """Load settings from the process environment, with project defaults."""
    load_dotenv()
    api_key = os.getenv("OPENAI_API_KEY", "").strip() or None
    return Settings(
        site=os.getenv("STACKEXCHANGE_SITE", "datascience").strip() or "datascience",
        pages=_positive_int(os.getenv("SCRAPE_PAGES"), 12),
        page_size=_positive_int(os.getenv("PAGE_SIZE"), 100),
        database_url=os.getenv("DATABASE_URL", DEFAULT_DATABASE_URL),
        openai_model=os.getenv("OPENAI_MODEL", "gpt-4o-mini"),
        openai_api_key=api_key,
    )
