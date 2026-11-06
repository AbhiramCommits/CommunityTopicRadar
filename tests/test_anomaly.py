"""Tests for robust-z and IsolationForest anomaly detection."""

from __future__ import annotations

import numpy as np
import pandas as pd

from ctr.analysis.anomaly import detect_anomalies, robust_anomalies


def _daily(values: list[float]) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "day": pd.date_range("2025-01-01", periods=len(values), freq="D", tz="UTC"),
            "posts_created": values,
            "answers_created": values,
            "unique_authors": values,
            "answered_count": values,
            "total_views": [value * 100 for value in values],
        }
    )


def test_injected_five_x_spike_is_flagged() -> None:
    values = [10.0] * 60
    values[40] = 50.0
    daily = _daily(values)
    robust = robust_anomalies(daily)
    assert not robust.empty
    assert pd.Timestamp("2025-02-10", tz="UTC") in set(robust["day"])
    anomalies, metrics = detect_anomalies(daily)
    assert anomalies["direction"].eq("spike").any()
    assert metrics["anomaly_count"] > 0


def test_flat_series_has_no_anomalies() -> None:
    daily = _daily([5.0] * 60)
    robust = robust_anomalies(daily)
    assert robust.empty
    anomalies, metrics = detect_anomalies(daily)
    assert anomalies.empty
    assert metrics["anomaly_count"] == 0


def test_drop_is_labelled() -> None:
    rng = np.random.default_rng(3)
    values = list(rng.normal(20, 1.0, size=60))
    values[45] = 1.0
    anomalies = robust_anomalies(_daily(values))
    assert (anomalies["direction"] == "drop").any()
