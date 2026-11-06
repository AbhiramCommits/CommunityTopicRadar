"""Engagement anomaly detection with robust z-scores and IsolationForest."""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest

logger = logging.getLogger(__name__)

METRIC_COLUMNS = [
    "posts_created",
    "answers_created",
    "unique_authors",
    "answered_count",
    "total_views",
]
ANOMALY_COLUMNS = [
    "day",
    "metric",
    "value",
    "expected",
    "deviation",
    "methods",
    "direction",
]


def _empty_anomalies() -> pd.DataFrame:
    return pd.DataFrame(columns=ANOMALY_COLUMNS)


def _robust_scale(values: pd.Series, median: pd.Series) -> pd.Series:
    mad = values.rolling(28, min_periods=7).apply(
        lambda window: float(np.median(np.abs(window - np.median(window)))), raw=True
    )
    scale = 1.4826 * mad
    std = values.rolling(28, min_periods=7).std()
    fallback = (median.abs() * 0.01).clip(lower=0.1)
    return scale.where(scale > 0, std).where(lambda series: series > 0, fallback)


def robust_anomalies(
    df: pd.DataFrame, metrics: list[str] | None = None, z_threshold: float = 3.0
) -> pd.DataFrame:
    """Flag days where a metric deviates more than 3 robust sigmas."""
    if df.empty:
        return _empty_anomalies()
    metrics = metrics or [column for column in METRIC_COLUMNS if column in df.columns]
    working = df.copy()
    working["day"] = pd.to_datetime(working["day"], utc=True, errors="coerce")
    working = working.sort_values("day").reset_index(drop=True)
    rows: list[dict[str, Any]] = []
    for metric in metrics:
        values = pd.to_numeric(working[metric], errors="coerce").astype(float)
        median = values.rolling(28, min_periods=7).median()
        scale = _robust_scale(values, median)
        deviation = values - median
        flagged = (scale.notna()) & (scale > 0) & (deviation.abs() > z_threshold * scale)
        for index in np.flatnonzero(flagged.to_numpy()):
            rows.append(
                {
                    "day": working.loc[index, "day"],
                    "metric": metric,
                    "value": float(values.iloc[index]),
                    "expected": float(median.iloc[index]),
                    "deviation": float(deviation.iloc[index]),
                    "methods": "robust_z",
                    "direction": "spike" if deviation.iloc[index] > 0 else "drop",
                }
            )
    result = pd.DataFrame(rows, columns=ANOMALY_COLUMNS)
    logger.info("Robust-z flagged %d metric-days", len(result))
    return result


def _day_flags(anomalies: pd.DataFrame) -> set[pd.Timestamp]:
    if anomalies.empty:
        return set()
    days: set[pd.Timestamp] = set()
    for value in anomalies["day"]:
        timestamp = pd.Timestamp(value)
        if timestamp.tzinfo is not None:
            timestamp = timestamp.tz_convert("UTC")
        days.add(timestamp.normalize())
    return days


def detect_anomalies(
    df: pd.DataFrame,
    contamination: float = 0.03,
    metrics: list[str] | None = None,
) -> tuple[pd.DataFrame, dict[str, float | int | None]]:
    """Combine robust-z flags with an IsolationForest and measure agreement."""
    if df.empty:
        return _empty_anomalies(), {
            "anomaly_count": 0,
            "anomaly_days": 0,
            "robust_z_days": 0,
            "isolation_forest_days": 0,
            "iforest_precision": None,
            "iforest_recall": None,
        }
    metrics = metrics or [column for column in METRIC_COLUMNS if column in df.columns]
    working = df.copy()
    working["day"] = pd.to_datetime(working["day"], utc=True, errors="coerce")
    working = working.sort_values("day").reset_index(drop=True)
    robust = robust_anomalies(working, metrics)
    rows: dict[tuple[pd.Timestamp, str], dict[str, Any]] = {}
    for record in robust.to_dict("records"):
        day = pd.Timestamp(record["day"])
        rows[(day.normalize(), record["metric"])] = record

    robust_days = _day_flags(robust)
    iforest_days: set[pd.Timestamp] = set()
    features = working[metrics].apply(pd.to_numeric, errors="coerce").fillna(0.0)
    if len(features) >= 20 and float(features.to_numpy().std()) > 0:
        model = IsolationForest(
            contamination=contamination,
            random_state=42,
            n_estimators=200,
        )
        predictions = model.fit_predict(features.to_numpy())
        for index in np.flatnonzero(predictions == -1):
            day = pd.Timestamp(working.loc[index, "day"]).normalize()
            iforest_days.add(day)
            deviations = {}
            for metric in metrics:
                values = pd.to_numeric(working[metric], errors="coerce").astype(float)
                median = values.rolling(28, min_periods=7).median().fillna(values.median())
                deviation = float(values.iloc[index] - median.iloc[index])
                deviations[metric] = (
                    deviation,
                    float(values.iloc[index]),
                    float(median.iloc[index]),
                )
            metric = max(deviations, key=lambda key: abs(deviations[key][0]))
            deviation, value, expected = deviations[metric]
            key = (day, metric)
            if key in rows:
                rows[key]["methods"] = "robust_z+isolation_forest"
            else:
                rows[key] = {
                    "day": day,
                    "metric": metric,
                    "value": value,
                    "expected": expected,
                    "deviation": deviation,
                    "methods": "isolation_forest",
                    "direction": "spike" if deviation > 0 else "drop",
                }

    true_positive = len(robust_days & iforest_days)
    precision = true_positive / len(iforest_days) if iforest_days else None
    recall = true_positive / len(robust_days) if robust_days else None
    result = pd.DataFrame(list(rows.values()), columns=ANOMALY_COLUMNS)
    if not result.empty:
        result = result.sort_values(["day", "metric"]).reset_index(drop=True)
    metrics_out: dict[str, float | int | None] = {
        "anomaly_count": int(len(result)),
        "anomaly_days": int(result["day"].nunique()) if not result.empty else 0,
        "robust_z_days": int(len(robust_days)),
        "isolation_forest_days": int(len(iforest_days)),
        "iforest_precision": float(precision) if precision is not None else None,
        "iforest_recall": float(recall) if recall is not None else None,
    }
    logger.info(
        "Anomaly detection: %d rows, %d days, IF precision=%s recall=%s",
        metrics_out["anomaly_count"],
        metrics_out["anomaly_days"],
        f"{precision:.3f}" if precision is not None else "n/a",
        f"{recall:.3f}" if recall is not None else "n/a",
    )
    return result, metrics_out
