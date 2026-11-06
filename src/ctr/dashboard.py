"""Executive Streamlit dashboard for CommunityTopicRadar."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

ROOT = Path(__file__).resolve().parents[2]
REPORTS = ROOT / "reports"
EXPORTS = ROOT / "exports"

st.set_page_config(
    page_title="CommunityTopicRadar",
    page_icon="📡",
    layout="wide",
    initial_sidebar_state="collapsed",
)


@st.cache_data(show_spinner=False)
def load_csv(path: str) -> pd.DataFrame:
    """Load a generated CSV artifact, returning an empty frame if absent."""
    file = Path(path)
    if not file.exists():
        return pd.DataFrame()
    return pd.read_csv(file)


def _metric_row() -> None:
    posts = load_csv(str(EXPORTS / "fact_post.csv"))
    authors = load_csv(str(EXPORTS / "dim_author.csv"))
    anomalies = load_csv(str(EXPORTS / "fact_anomaly.csv"))
    gaps = load_csv(str(REPORTS / "content_gaps.csv"))
    if posts.empty:
        st.error("No exports found. Run `python -m ctr.cli export` first.")
        return
    answer_rate = float(posts["is_answered"].mean()) if "is_answered" in posts else 0.0
    columns = st.columns(5)
    columns[0].metric("Posts", f"{len(posts):,}")
    columns[1].metric("Authors", f"{len(authors):,}")
    columns[2].metric("Answer rate", f"{answer_rate:.1%}")
    columns[3].metric("Content-gap topics", f"{len(gaps):,}")
    anomaly_days = anomalies["date_key"].nunique() if not anomalies.empty else 0
    columns[4].metric("Anomaly days", f"{anomaly_days:,}")


def _reach_tab() -> None:
    daily = load_csv(str(EXPORTS / "fact_engagement_daily.csv"))
    posts = load_csv(str(EXPORTS / "fact_post.csv"))
    if daily.empty or posts.empty:
        st.info("No engagement export found.")
        return
    daily = daily.copy()
    daily["date"] = pd.to_datetime(daily["date_key"].astype(str), format="%Y%m%d")
    daily = daily.sort_values("date")
    posts = posts.copy()
    posts["date"] = pd.to_datetime(posts["date_key"].astype(str), format="%Y%m%d")

    fig = make_subplots(specs=[[{"secondary_y": True}]])
    fig.add_trace(
        go.Scatter(
            x=daily["date"],
            y=daily["posts_created"],
            mode="lines",
            name="Posts created",
        ),
        secondary_y=False,
    )
    fig.add_trace(
        go.Scatter(
            x=daily["date"],
            y=daily["unique_authors"],
            mode="lines",
            name="Active authors",
        ),
        secondary_y=False,
    )
    fig.add_trace(
        go.Scatter(
            x=daily["date"],
            y=daily["total_views"],
            mode="lines",
            name="Views",
            opacity=0.45,
        ),
        secondary_y=True,
    )
    fig.update_layout(
        title="Posts, active authors, and views over time",
        hovermode="x unified",
    )
    fig.update_yaxes(title_text="Posts / active authors", secondary_y=False)
    fig.update_yaxes(title_text="Views", secondary_y=True, showgrid=False)
    st.plotly_chart(fig, use_container_width=True)

    month = posts["date"].dt.to_period("M").dt.to_timestamp()
    monthly_posts = (
        posts.assign(month=month)
        .groupby("month")
        .agg(posts_created=("post_id", "count"), unique_authors=("author_id", "nunique"))
    )
    monthly_daily = (
        daily.assign(month=daily["date"].dt.to_period("M").dt.to_timestamp())
        .groupby("month")
        .agg(answers_created=("answers_created", "sum"), total_views=("total_views", "sum"))
    )
    monthly = monthly_posts.join(monthly_daily, how="outer").fillna(0).sort_index()
    monthly["posts_mom_growth"] = monthly["posts_created"].pct_change() * 100

    current_period = daily["date"].max().to_period("M")
    complete = monthly[monthly.index.to_period("M") < current_period]
    if len(complete) >= 1:
        current = complete.iloc[-1]
        columns = st.columns(3)
        columns[0].metric(
            "Latest complete month posts",
            f"{int(current['posts_created']):,}",
            f"{current['posts_mom_growth']:+.1f}% MoM",
        )
        columns[1].metric(
            "Latest complete month authors", f"{int(current['unique_authors']):,}"
        )
        columns[2].metric(
            "Latest complete month answers", f"{int(current['answers_created']):,}"
        )
    st.subheader("Monthly reach")
    st.dataframe(monthly.tail(12), use_container_width=True)


def _trending_tab() -> None:
    trending = load_csv(str(REPORTS / "trending_topics.csv"))
    if trending.empty:
        st.info("No trending analysis found. Run `python -m ctr.cli analyze`.")
        return
    top = trending.head(20).copy()
    fig = px.bar(
        top.sort_values("z_score"),
        x="z_score",
        y="term",
        color="kind",
        orientation="h",
        title="Week-over-week movers by z-score",
        hover_data=["current_count", "prior_count", "pct_change"],
    )
    st.plotly_chart(fig, use_container_width=True)
    st.subheader("Trending table")
    st.dataframe(top, use_container_width=True)


def _gaps_tab() -> None:
    gaps = load_csv(str(REPORTS / "content_gaps.csv"))
    if gaps.empty:
        st.info("No content-gap analysis found. Run `python -m ctr.cli analyze`.")
        return
    fig = px.scatter(
        gaps,
        x="answer_rate",
        y="gap_score",
        size="posts",
        color="median_views",
        hover_name="topic",
        title="Content gaps: low answer rate and high demand",
        labels={"answer_rate": "Answer rate", "gap_score": "Gap score"},
    )
    st.plotly_chart(fig, use_container_width=True)
    st.subheader("Ranked backlog")
    st.dataframe(
        gaps,
        use_container_width=True,
        column_config={
            "example_post_url": st.column_config.LinkColumn(
                "Example post", display_text="Open unanswered example"
            )
        },
    )


def _segments_tab() -> None:
    authors = load_csv(str(EXPORTS / "dim_author.csv"))
    if authors.empty or "pca_x" not in authors:
        st.info("No segment export found.")
        return
    plot_frame = authors.dropna(subset=["pca_x", "pca_y"]).copy()
    fig = px.scatter(
        plot_frame,
        x="pca_x",
        y="pca_y",
        color="segment",
        hover_data=["author_id", "posts_asked", "answers_given"],
        title="Author segments on the first two PCA components",
        opacity=0.65,
    )
    fig.update_traces(marker={"size": 7})
    st.plotly_chart(fig, use_container_width=True)
    sizes = (
        authors.groupby("segment")
        .agg(
            authors=("author_id", "count"),
            posts_asked=("posts_asked", "mean"),
            answers_given=("answers_given", "mean"),
            active_days=("active_days", "mean"),
        )
        .reset_index()
        .sort_values("authors", ascending=False)
    )
    st.subheader("Segment sizes")
    st.dataframe(sizes, use_container_width=True)


def _anomalies_tab() -> None:
    daily = load_csv(str(EXPORTS / "fact_engagement_daily.csv"))
    anomalies = load_csv(str(EXPORTS / "fact_anomaly.csv"))
    if daily.empty:
        st.info("No engagement export found.")
        return
    daily = daily.copy()
    daily["date"] = pd.to_datetime(daily["date_key"].astype(str), format="%Y%m%d")
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=daily["date"],
            y=daily["posts_created"],
            mode="lines",
            name="Posts created",
        )
    )
    if not anomalies.empty:
        anomaly = anomalies.copy()
        anomaly["date"] = pd.to_datetime(anomaly["date_key"].astype(str), format="%Y%m%d")
        flagged = daily[daily["is_anomaly"].astype(bool)]
        fig.add_trace(
            go.Scatter(
                x=flagged["date"],
                y=flagged["posts_created"],
                mode="markers",
                marker={"color": "crimson", "size": 10},
                name="Flagged day",
            )
        )
        st.subheader("Flagged days")
        st.dataframe(anomaly, use_container_width=True)
    fig.update_layout(title="Engagement timeline with anomalous days")
    st.plotly_chart(fig, use_container_width=True)


st.title("CommunityTopicRadar")
st.caption("Community-engagement analytics for program owners")
_metric_row()
reach, trending, gaps, segments, anomalies = st.tabs(
    ["Reach & Growth", "Trending Topics", "Content Gaps", "Community Segments", "Anomalies"]
)
with reach:
    _reach_tab()
with trending:
    _trending_tab()
with gaps:
    _gaps_tab()
with segments:
    _segments_tab()
with anomalies:
    _anomalies_tab()
