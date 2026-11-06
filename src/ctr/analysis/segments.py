"""Author engagement features, K-Means segments, and DBSCAN comparison."""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd
from sklearn.cluster import DBSCAN, KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler

logger = logging.getLogger(__name__)

FEATURE_COLUMNS = [
    "posts_asked",
    "answers_given",
    "accepted_answers",
    "avg_score",
    "median_views",
    "active_days",
    "answer_to_ask_ratio",
    "recency_days",
]
RANDOM_STATE = 42


def _empty_features() -> pd.DataFrame:
    return pd.DataFrame(columns=["author_id", *FEATURE_COLUMNS])


def build_author_features(posts_df: pd.DataFrame, answers_df: pd.DataFrame) -> pd.DataFrame:
    """Build one activity profile per author from questions and answers."""
    post_frames: list[pd.DataFrame] = []
    answer_frames: list[pd.DataFrame] = []
    activity_frames: list[pd.DataFrame] = []

    if not posts_df.empty and "author_id" in posts_df.columns:
        posts = posts_df.dropna(subset=["author_id"]).copy()
        posts["author_id"] = posts["author_id"].astype(int)
        posts["created_at"] = pd.to_datetime(posts["created_at"], utc=True, errors="coerce")
        post_stats = posts.groupby("author_id").agg(
            posts_asked=("id", "count"),
            avg_score=("score", "mean"),
            median_views=("view_count", "median"),
        )
        post_frames.append(post_stats)
        activity_frames.append(
            posts[["author_id", "created_at"]].rename(columns={"created_at": "active_at"})
        )

    if not answers_df.empty and "author_id" in answers_df.columns:
        answers = answers_df.dropna(subset=["author_id"]).copy()
        answers["author_id"] = answers["author_id"].astype(int)
        answers["created_at"] = pd.to_datetime(answers["created_at"], utc=True, errors="coerce")
        answer_stats = answers.groupby("author_id").agg(
            answers_given=("id", "count"),
            accepted_answers=("is_accepted", "sum"),
        )
        answer_frames.append(answer_stats)
        activity_frames.append(
            answers[["author_id", "created_at"]].rename(columns={"created_at": "active_at"})
        )

    if not post_frames and not answer_frames:
        return _empty_features()

    features = pd.concat(post_frames + answer_frames, axis=1)
    features = features.fillna(
        {
            "posts_asked": 0,
            "avg_score": 0.0,
            "median_views": 0.0,
            "answers_given": 0,
            "accepted_answers": 0,
        }
    )
    for column in ("posts_asked", "answers_given", "accepted_answers"):
        features[column] = features[column].astype(int)
    features["answer_to_ask_ratio"] = features["answers_given"] / features["posts_asked"].clip(
        lower=1
    )

    activity = pd.concat(activity_frames, ignore_index=True).dropna(subset=["active_at"])
    if not activity.empty:
        activity_days = activity.groupby("author_id")["active_at"].apply(
            lambda values: values.dt.date.nunique()
        )
        last_seen = activity.groupby("author_id")["active_at"].max()
        newest = activity["active_at"].max()
        features["active_days"] = activity_days.reindex(features.index).fillna(0).astype(int)
        features["recency_days"] = (
            newest - last_seen.reindex(features.index)
        ).dt.total_seconds().div(86_400).fillna(0).astype(int)
    else:
        features["active_days"] = 0
        features["recency_days"] = 0

    features = features.fillna(0.0)
    features = features.reset_index().rename(columns={"index": "author_id"})
    features["author_id"] = features["author_id"].astype(int)
    return features[["author_id", *FEATURE_COLUMNS]]


def label_segments(centroids: pd.DataFrame) -> list[str]:
    """Map cluster centroids to lurker/asker/answerer/champion names.

    The champion is the cluster with the strongest answer activity. Other
    clusters are labelled by their mix of answering and asking: contributors
    with meaningful answer activity are answerers, multi-question authors are
    askers, and low-activity one-question clusters are lurkers.
    """
    centroids = centroids.copy()
    if centroids.empty:
        return []
    if len(centroids) == 1:
        row = centroids.iloc[0]
        if row.get("answers_given", 0) >= 1:
            return ["champion"]
        if row.get("posts_asked", 0) >= 2:
            return ["asker"]
        return ["lurker"]

    answer_index = (
        centroids["answers_given"]
        + 0.4 * centroids["accepted_answers"]
        + 0.5 * centroids["answer_to_ask_ratio"]
    )
    labels: dict[object, str] = {}
    if float(centroids["answers_given"].max()) >= 1:
        champion = answer_index.idxmax()
        labels[champion] = "champion"
    for cluster, row in centroids.iterrows():
        if cluster in labels:
            continue
        if row["answers_given"] >= 1 and row["answer_to_ask_ratio"] >= 0.5:
            labels[cluster] = "answerer"
        elif row["posts_asked"] >= 2:
            labels[cluster] = "asker"
        else:
            labels[cluster] = "lurker"
    return [labels[cluster] for cluster in centroids.index]


def _dbscan_silhouette(scaled: np.ndarray) -> tuple[float | None, int, int]:
    best_score: float | None = None
    best_clusters = 0
    best_noise = 0
    for eps in (0.6, 0.9, 1.2, 1.5, 1.8, 2.2):
        labels = DBSCAN(eps=eps, min_samples=5).fit_predict(scaled)
        core_mask = labels != -1
        clusters = set(labels[core_mask].tolist())
        if len(clusters) < 2 or len(clusters) >= len(scaled) - 1:
            continue
        if core_mask.sum() <= len(clusters):
            continue
        try:
            score = float(silhouette_score(scaled[core_mask], labels[core_mask]))
        except ValueError:
            continue
        if best_score is None or score > best_score:
            best_score = score
            best_clusters = len(clusters)
            best_noise = int((~core_mask).sum())
    logger.info(
        "DBSCAN comparison: silhouette=%s clusters=%d noise=%d",
        f"{best_score:.3f}" if best_score is not None else "n/a",
        best_clusters,
        best_noise,
    )
    return best_score, best_clusters, best_noise


def cluster_authors(
    features: pd.DataFrame,
    min_k: int = 3,
    max_k: int = 8,
) -> tuple[pd.DataFrame, dict[str, float | int | None]]:
    """Cluster authors, selecting K-Means k by silhouette, and compare DBSCAN.

    The returned frame includes cluster ids, human labels, and two PCA
    coordinates for the dashboard scatter plot.
    """
    if features.empty:
        return features.copy(), {
            "kmeans_k": None,
            "kmeans_silhouette": None,
            "dbscan_silhouette": None,
            "dbscan_clusters": 0,
            "dbscan_noise": 0,
        }
    scalar = StandardScaler()
    scaled = scalar.fit_transform(features[FEATURE_COLUMNS].to_numpy())
    n_samples = len(features)
    upper = min(max_k, n_samples - 1)
    lower = min(min_k, upper)
    best_k = lower
    best_score: float | None = None
    best_labels: np.ndarray | None = None
    for candidate in range(lower, upper + 1):
        if candidate < 2:
            continue
        model = KMeans(n_clusters=candidate, n_init=10, random_state=RANDOM_STATE)
        labels = model.fit_predict(scaled)
        if len(set(labels)) < 2:
            continue
        score = float(silhouette_score(scaled, labels))
        logger.info("KMeans k=%d silhouette=%.3f", candidate, score)
        if best_score is None or score > best_score:
            best_k = candidate
            best_score = score
            best_labels = labels
    if best_labels is None:
        best_k = min(2, max(1, n_samples - 1))
        best_labels = KMeans(
            n_clusters=best_k, n_init=10, random_state=RANDOM_STATE
        ).fit_predict(scaled)

    result = features.copy()
    result["cluster"] = best_labels
    centroids = result.groupby("cluster")[FEATURE_COLUMNS].mean()
    labels = label_segments(centroids)
    label_map = dict(zip(centroids.index, labels, strict=False))
    result["segment"] = result["cluster"].map(label_map)
    pca = PCA(n_components=2, random_state=RANDOM_STATE)
    components = pca.fit_transform(scaled)
    result["pca_x"] = components[:, 0]
    result["pca_y"] = components[:, 1]

    dbscan_score, dbscan_clusters, dbscan_noise = _dbscan_silhouette(scaled)
    metrics: dict[str, float | int | None] = {
        "kmeans_k": int(best_k),
        "kmeans_silhouette": best_score,
        "dbscan_silhouette": dbscan_score,
        "dbscan_clusters": dbscan_clusters,
        "dbscan_noise": dbscan_noise,
    }
    logger.info("Selected K-Means k=%d silhouette=%.3f", best_k, best_score or 0.0)
    return result, metrics
