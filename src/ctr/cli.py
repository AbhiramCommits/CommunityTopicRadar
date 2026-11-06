"""Command-line entry point for CommunityTopicRadar."""

from __future__ import annotations

import argparse
import logging
import sys
from collections.abc import Sequence
from dataclasses import replace
from pathlib import Path

from tabulate import tabulate

from ctr.config import Settings, load_settings
from ctr.ingest import run_ingest

logger = logging.getLogger(__name__)


def build_parser() -> argparse.ArgumentParser:
    """Build the CLI argument parser."""
    parser = argparse.ArgumentParser(
        prog="ctr",
        description="CommunityTopicRadar analytics pipeline",
    )
    parser.add_argument("-v", "--verbose", action="store_true", help="enable debug logging")
    subparsers = parser.add_subparsers(dest="command", required=True)

    ingest = subparsers.add_parser("ingest", help="scrape or replay data into the warehouse")
    ingest.add_argument(
        "--offline",
        action="store_true",
        help="replay data/raw (or the committed sample) without network access",
    )
    ingest.add_argument("--sample", type=Path, help="ingest a JSON sample instead")
    ingest.add_argument("--database-url", help="override DATABASE_URL for this command")
    ingest.set_defaults(handler=handle_ingest)

    analyze = subparsers.add_parser("analyze", help="run all analytics and write reports")
    analyze.add_argument("--database-url", help="override DATABASE_URL for this command")
    analyze.set_defaults(handler=handle_analyze)

    brief = subparsers.add_parser("brief", help="generate plain-language topic briefs")
    brief.add_argument("--database-url", help="override DATABASE_URL for this command")
    brief.set_defaults(handler=handle_brief)

    export = subparsers.add_parser("export", help="write the Power BI star schema")
    export.add_argument("--database-url", help="override DATABASE_URL for this command")
    export.set_defaults(handler=handle_export)

    dashboard = subparsers.add_parser("dashboard", help="show the Streamlit launch command")
    dashboard.set_defaults(handler=handle_dashboard)
    return parser


def _settings_with_override(args: argparse.Namespace) -> Settings:
    settings = load_settings()
    database_url = getattr(args, "database_url", None)
    return replace(settings, database_url=database_url) if database_url else settings


def handle_ingest(args: argparse.Namespace) -> int:
    """Run ingestion and print row counts."""
    settings = _settings_with_override(args)
    stats = run_ingest(
        settings,
        offline=args.offline,
        sample_path=args.sample,
    )
    rows = [
        ["posts", f"{stats['posts']:,}"],
        ["answers", f"{stats['answers']:,}"],
        ["authors", f"{stats['authors']:,}"],
        ["tags", f"{stats['tags']:,}"],
        ["post_tags", f"{stats['post_tags']:,}"],
        ["engagement_daily", f"{stats['engagement_daily']:,}"],
    ]
    print(tabulate(rows, headers=["table", "rows"], tablefmt="github"))
    print(
        f"\nSource: {stats['source']} ({stats['source_site']}), "
        f"date range {str(stats['date_min'])[:10]} to {str(stats['date_max'])[:10]}, "
        f"{stats['run_seconds']}s"
    )
    return 0


def handle_analyze(args: argparse.Namespace) -> int:
    """Run the analytics suite (implemented in the analytics step)."""
    from ctr.analysis.pipeline import run_analysis

    settings = _settings_with_override(args)
    summary = run_analysis(settings)
    print(summary)
    return 0


def handle_brief(args: argparse.Namespace) -> int:
    """Generate topic briefs (implemented in the briefing step)."""
    from ctr.brief import run_briefs

    settings = _settings_with_override(args)
    output = run_briefs(settings)
    print(f"Wrote {output}")
    return 0


def handle_export(args: argparse.Namespace) -> int:
    """Write the star schema (implemented in the export step)."""
    from ctr.export import run_export

    settings = _settings_with_override(args)
    outputs = run_export(settings)
    print(f"Wrote {len(outputs)} export files to {settings.exports_dir}")
    return 0


def handle_dashboard(args: argparse.Namespace) -> int:
    """Print the dashboard launch command."""
    print("Run: streamlit run src/ctr/dashboard.py")
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    """Parse arguments and dispatch the requested command."""
    parser = build_parser()
    args = parser.parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    try:
        return int(args.handler(args))
    except Exception:
        logger.exception("Command failed")
        return 1


if __name__ == "__main__":
    sys.exit(main())
