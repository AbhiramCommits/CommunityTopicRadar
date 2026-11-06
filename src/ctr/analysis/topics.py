"""Trending-term detection and NMF topic modelling."""

from __future__ import annotations

import logging
import math
from typing import Any

import numpy as np
import pandas as pd
from sklearn.decomposition import NMF
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS, CountVectorizer, TfidfVectorizer

logger = logging.getLogger(__name__)

LATEX_STOP_WORDS = {
    "alpha",
    "approx",
    "beta",
    "begin",
    "cdot",
    "data",
    "df",
    "end",
    "exp",
    "family",
    "frac",
    "ge",
    "hat",
    "infty",
    "int",
    "intercept",
    "lambda",
    "ldots",
    "left",
    "le",
    "library",
    "log",
    "mathcal",
    "mathbb",
    "mathrm",
    "mathbf",
    "mu",
    "neq",
    "partial",
    "prime",
    "pr",
    "right",
    "sigma",
    "sqrt",
    "std",
    "sum",
    "summary",
    "text",
    "theta",
    "tilde",
    "times",
}
STOP_WORDS = sorted(set(ENGLISH_STOP_WORDS) | LATEX_STOP_WORDS)
TFIDF_PARAMS: dict[str, Any] = {
    "stop_words": STOP_WORDS,
    "ngram_range": (1, 2),
    "min_df": 5,
    "max_features": 4000,
    "sublinear_tf": True,
    "token_pattern": r"(?u)\b[a-zA-Z][a-zA-Z]+\b",
}
RANDOM_STATE = 42
BOILERPLATE_TERMS = {
    "anyone",
    "approach",
    "based",
    "best",
    "calculate",
    "calculation",
    "case",
    "cases",
    "data",
    "difference",
    "different",
    "example",
    "examples",
    "explain",
    "false",
    "good",
    "help",
    "high",
    "interpret",
    "interpretation",
    "know",
    "like",
    "make",
    "many",
    "method",
    "methods",
    "much",
    "need",
    "one",
    "please",
    "point",
    "possible",
    "problem",
    "problems",
    "question",
    "questions",
    "result",
    "results",
    "someone",
    "still",
    "thank",
    "thanks",
    "three",
    "time",
    "times",
    "true",
    "two",
    "understand",
    "understanding",
    "use",
    "used",
    "using",
    "value",
    "values",
    "want",
    "way",
    "work",
    "works",
}


def _documents(df: pd.DataFrame) -> pd.Series:
    title = df.get("title", pd.Series("", index=df.index)).fillna("").astype(str)
    body = df.get("body_text", pd.Series("", index=df.index)).fillna("").astype(str)
    return (title + " " + body).str.strip()


def _utc_series(df: pd.DataFrame) -> pd.Series:
    return pd.to_datetime(df["created_at"], utc=True, errors="coerce")


def _window_bounds(df: pd.DataFrame, window_days: int) -> tuple[pd.Timestamp, ...]:
    dates = _utc_series(df)
    if dates.dropna().empty:
        now = pd.Timestamp.now(tz="UTC").normalize()
        end = now - pd.Timedelta(days=1)
    else:
        end = dates.max().normalize() - pd.Timedelta(days=1)
    current_start = end - pd.Timedelta(days=window_days - 1)
    prior_end = current_start - pd.Timedelta(days=1)
    prior_start = prior_end - pd.Timedelta(days=window_days - 1)
    return current_start, end, prior_start, prior_end


def _two_proportion_z(current: float, prior: float, n_current: int, n_prior: int) -> float:
    if n_current <= 0 or n_prior <= 0:
        return 0.0
    p_current = current / n_current
    p_prior = prior / n_prior
    pooled = (current + prior) / (n_current + n_prior)
    variance = pooled * (1 - pooled) * (1 / n_current + 1 / n_prior)
    if variance <= 0:
        return 0.0
    return float((p_current - p_prior) / math.sqrt(variance))


def _pct_change(current: float, prior: float) -> float:
    if prior > 0:
        return float((current - prior) / prior * 100.0)
    return 100.0 if current > 0 else 0.0


def _counts_for_series(
    series: pd.Series, dates: pd.Series, bounds: tuple[Any, ...]
) -> tuple[pd.Series, pd.Series]:
    current_start, current_end, prior_start, prior_end = bounds
    current = series[(dates >= current_start) & (dates <= current_end)].value_counts()
    prior = series[(dates >= prior_start) & (dates <= prior_end)].value_counts()
    return current, prior


def rank_trending(df: pd.DataFrame, window_days: int = 7) -> pd.DataFrame:
    """Rank tags and title/body keywords by week-over-week volume change.

    The latest complete week is the seven full days before the most recent
    observed day. The comparison week is the seven days immediately before it.
    """
    empty = pd.DataFrame(
        columns=[
            "term",
            "current_count",
            "prior_count",
            "delta",
            "pct_change",
            "z_score",
            "kind",
        ]
    )
    if df.empty:
        return empty
    working = df.copy()
    working["_date"] = _utc_series(working)
    bounds = _window_bounds(working, window_days)
    current_start, current_end, prior_start, prior_end = bounds
    n_current = int(((working["_date"] >= current_start) & (working["_date"] <= current_end)).sum())
    n_prior = int(((working["_date"] >= prior_start) & (working["_date"] <= prior_end)).sum())

    frames: list[pd.DataFrame] = []
    if "tags" in working.columns:
        tag_frame = working[["_date", "tags"]].explode("tags").dropna(subset=["tags"])
        tag_frame["tags"] = tag_frame["tags"].astype(str)
        if not tag_frame.empty:
            current, prior = _counts_for_series(tag_frame["tags"], tag_frame["_date"], bounds)
            frames.append(_trend_frame(current, prior, n_current, n_prior, "tag"))

    documents = _documents(working)
    if documents.str.len().sum() > 0:
        vectorizer = CountVectorizer(
            stop_words=STOP_WORDS,
            ngram_range=(1, 2),
            min_df=2,
            max_features=3000,
            binary=True,
            token_pattern=r"(?u)\b[a-zA-Z]{3,}\b",
        )
        matrix = vectorizer.fit_transform(documents)
        terms = np.asarray(vectorizer.get_feature_names_out())
        current_mask = (
            (working["_date"] >= current_start) & (working["_date"] <= current_end)
        ).to_numpy()
        prior_mask = (
            (working["_date"] >= prior_start) & (working["_date"] <= prior_end)
        ).to_numpy()
        current_counts = np.asarray(matrix[current_mask].sum(axis=0)).ravel()
        prior_counts = np.asarray(matrix[prior_mask].sum(axis=0)).ravel()
        keyword_frame = pd.DataFrame(
            {"term": terms, "current_count": current_counts, "prior_count": prior_counts}
        )
        keyword_frame = keyword_frame[
            (keyword_frame["current_count"] + keyword_frame["prior_count"]) >= 2
        ]
        keyword_frame = keyword_frame[
            keyword_frame["term"]
            .str.split()
            .apply(lambda words: all(word not in BOILERPLATE_TERMS for word in words))
        ]
        frames.append(_trend_frame_from_rows(keyword_frame, n_current, n_prior, "keyword"))

    if not frames:
        return empty
    result = pd.concat(frames, ignore_index=True)
    result["z_score"] = result["z_score"].replace([np.inf, -np.inf], 0.0).fillna(0.0)
    result = result.sort_values(
        ["z_score", "current_count", "delta"], ascending=False
    ).reset_index(drop=True)
    return result[empty.columns]


def _trend_frame(
    current: pd.Series,
    prior: pd.Series,
    n_current: int,
    n_prior: int,
    kind: str,
) -> pd.DataFrame:
    terms = sorted(set(current.index).union(prior.index))
    if not terms:
        return pd.DataFrame(
            columns=[
                "term",
                "current_count",
                "prior_count",
                "delta",
                "pct_change",
                "z_score",
                "kind",
            ]
        )
    rows = [
        {
            "term": term,
            "current_count": int(current.get(term, 0)),
            "prior_count": int(prior.get(term, 0)),
            "kind": kind,
        }
        for term in terms
    ]
    return _trend_frame_from_rows(pd.DataFrame(rows), n_current, n_prior, kind)


def _trend_frame_from_rows(
    frame: pd.DataFrame, n_current: int, n_prior: int, kind: str
) -> pd.DataFrame:
    result = frame.copy()
    result["kind"] = kind
    result["delta"] = result["current_count"] - result["prior_count"]
    result["pct_change"] = [
        _pct_change(float(current), float(prior))
        for current, prior in zip(result["current_count"], result["prior_count"], strict=False)
    ]
    result["z_score"] = [
        _two_proportion_z(float(current), float(prior), n_current, n_prior)
        for current, prior in zip(result["current_count"], result["prior_count"], strict=False)
    ]
    return result[
        ["term", "current_count", "prior_count", "delta", "pct_change", "z_score", "kind"]
    ]


def _elbow_index(values: np.ndarray) -> int:
    if len(values) == 1:
        return 0
    x = np.linspace(0, 1, len(values))
    y = (values - values.min()) / max(values.max() - values.min(), 1e-12)
    start = np.array([x[0], y[0]])
    end = np.array([x[-1], y[-1]])
    line = end - start
    norm = np.linalg.norm(line) or 1.0
    distances = [
        abs(line[0] * (point_y - start[1]) - line[1] * (point_x - start[0])) / norm
        for point_x, point_y in zip(x, y, strict=False)
    ]
    return int(np.argmax(distances))


def topic_model(
    df: pd.DataFrame, k: int | None = None
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Fit NMF topics and return top terms and document-topic assignments.

    With ``k=None``, candidate counts from 6 through 14 are evaluated and the
    reconstruction-error elbow is selected. ``topics.attrs`` carries the
    chosen k and error curve for reporting.
    """
    documents = _documents(df)
    if documents.empty or documents.str.len().sum() == 0:
        return pd.DataFrame(columns=["topic", "rank", "term", "weight"]), pd.DataFrame(
            columns=["id", "topic", "weight"]
        )
    vectorizer = TfidfVectorizer(**TFIDF_PARAMS)
    matrix = vectorizer.fit_transform(documents)
    terms = np.asarray(vectorizer.get_feature_names_out())
    if matrix.shape[1] < 2:
        return pd.DataFrame(columns=["topic", "rank", "term", "weight"]), pd.DataFrame(
            columns=["id", "topic", "weight"]
        )

    max_possible = min(14, matrix.shape[0] - 1, matrix.shape[1])
    min_k = min(6, max_possible)
    candidates = list(range(min_k, max_possible + 1))
    errors: dict[int, float] = {}
    fitted: dict[int, NMF] = {}
    if k is None and len(candidates) > 1:
        for candidate in candidates:
            model = NMF(
                n_components=candidate,
                init="nndsvda",
                random_state=RANDOM_STATE,
                max_iter=250,
                solver="cd",
            )
            weights = model.fit_transform(matrix)
            fitted[candidate] = model
            errors[candidate] = float(model.reconstruction_err_)
            logger.info("NMF k=%d reconstruction error=%.4f", candidate, errors[candidate])
        chosen = candidates[_elbow_index(np.array([errors[c] for c in candidates]))]
    else:
        chosen = int(k or min_k)
        chosen = max(2, min(chosen, max_possible))
        model = NMF(
            n_components=chosen,
            init="nndsvda",
            random_state=RANDOM_STATE,
            max_iter=250,
            solver="cd",
        )
        model.fit(matrix)
        fitted[chosen] = model
        errors[chosen] = float(model.reconstruction_err_)

    model = fitted.get(chosen)
    if model is None:
        model = NMF(
            n_components=chosen,
            init="nndsvda",
            random_state=RANDOM_STATE,
            max_iter=250,
            solver="cd",
        )
        model.fit(matrix)
        fitted[chosen] = model
        errors[chosen] = float(model.reconstruction_err_)

    components = model.components_
    topic_rows = []
    for topic_index, component in enumerate(components):
        top_indices = np.argsort(component)[::-1][:10]
        for rank, term_index in enumerate(top_indices, start=1):
            topic_rows.append(
                {
                    "topic": topic_index,
                    "rank": rank,
                    "term": str(terms[term_index]),
                    "weight": float(component[term_index]),
                }
            )
    topics = pd.DataFrame(topic_rows)
    weights = model.transform(matrix)
    assignments = pd.DataFrame(
        {
            "id": df.get("id", pd.Series(range(len(df)))).to_numpy(),
            "topic": weights.argmax(axis=1),
            "weight": weights.max(axis=1),
        }
    )
    topics.attrs["k"] = chosen
    topics.attrs["reconstruction_error"] = errors.get(chosen)
    topics.attrs["k_errors"] = errors
    logger.info("Chose NMF k=%d by reconstruction-error elbow", chosen)
    return topics, assignments
