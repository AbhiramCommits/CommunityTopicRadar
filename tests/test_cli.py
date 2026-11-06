"""Tests for CLI parsing and command dispatch."""

from __future__ import annotations

from pathlib import Path

from ctr import brief, cli, export
from ctr.analysis import pipeline


def _stats() -> dict:
    return {
        "posts": 2,
        "answers": 1,
        "authors": 2,
        "tags": 3,
        "post_tags": 4,
        "engagement_daily": 1,
        "source": "sample",
        "source_site": "stats",
        "date_min": "2025-01-01T00:00:00",
        "date_max": "2025-01-02T00:00:00",
        "run_seconds": 0.1,
    }


def test_build_parser_has_commands() -> None:
    parser = cli.build_parser()
    for command in ("ingest", "analyze", "brief", "export", "dashboard"):
        assert command in parser.format_help()


def test_dashboard_and_analysis_dispatch(settings, monkeypatch, capsys) -> None:
    monkeypatch.setattr(cli, "load_settings", lambda: settings)
    monkeypatch.setattr(cli, "run_ingest", lambda *args, **kwargs: _stats())
    assert cli.main(["ingest", "--offline"]) == 0
    assert "6,500" not in capsys.readouterr().out
    monkeypatch.setattr(pipeline, "run_analysis", lambda active: "summary")
    assert cli.main(["analyze"]) == 0
    assert "summary" in capsys.readouterr().out
    monkeypatch.setattr(brief, "run_briefs", lambda active: Path("briefs.md"))
    assert cli.main(["brief"]) == 0
    monkeypatch.setattr(export, "run_export", lambda active: [Path("a"), Path("b")])
    assert cli.main(["export"]) == 0
    assert cli.main(["dashboard"]) == 0


def test_main_reports_failure(settings, monkeypatch) -> None:
    monkeypatch.setattr(cli, "load_settings", lambda: settings)

    def fail(*args, **kwargs):
        raise RuntimeError("boom")

    monkeypatch.setattr(cli, "run_ingest", fail)
    assert cli.main(["ingest", "--offline"]) == 1


def test_database_url_override(settings, monkeypatch) -> None:
    monkeypatch.setattr(cli, "load_settings", lambda: settings)
    parser = cli.build_parser()
    args = parser.parse_args(["ingest", "--database-url", "sqlite:///x.db"])
    assert cli._settings_with_override(args).database_url == "sqlite:///x.db"
