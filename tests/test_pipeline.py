"""End-to-end analytics pipeline test on the committed sample."""

from __future__ import annotations

import json

from tests.conftest import SAMPLE_POSTS

from ctr.analysis.pipeline import run_analysis
from ctr.ingest import run_ingest


def test_analysis_pipeline_on_sample(settings) -> None:
    run_ingest(
        settings,
        offline=True,
        sample_path=SAMPLE_POSTS,
        write_stats=False,
    )
    summary = run_analysis(settings)
    assert "Posts analyzed" in summary
    metrics = json.loads((settings.reports_dir / "metrics.json").read_text())
    assert metrics["posts_analyzed"] == 200
    assert metrics["answers_analyzed"] == 184
    assert metrics["nmf_k"] >= 2
    assert metrics["kmeans_k"] >= 3
    assert metrics["kmeans_silhouette"] is not None
    assert metrics["anomaly_count"] is not None
    for name in (
        "content_gaps.csv",
        "segments.csv",
        "segment_profile.md",
        "anomalies.csv",
        "topics.csv",
        "topic_assignments.csv",
        "trending_topics.csv",
    ):
        assert (settings.reports_dir / name).exists()
