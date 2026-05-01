"""
Per-User Sentiment Analysis page for the Maya dashboard.
"""

from __future__ import annotations

from typing import Dict
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from apps.dashboard.shared import (
    SENTIMENT_COLORS, SENTIMENT_DIVERGING_SCALE, ACCENT_PRIMARY,
    RISK_COLORS, style_chart, executive_card, executive_metric,
    humanize_feature_name, shorten_user_label,
)
from apps.dashboard.data_loaders import (
    load_sentiment_table,
    build_task_importance,
    build_latest_interaction_scores,
    build_response_sentiment_timeline,
    build_representative_statements,
    load_sentiment_table,
    build_task_importance,
    build_latest_interaction_scores,
    build_response_sentiment_timeline,
    build_representative_statements,
)


def render(wa: pd.DataFrame, user_directory: pd.DataFrame, name_map: Dict[int, str], refresh_nonce: str) -> None:
    """Render the Per-User Sentiment Analysis page."""



    per_user = wa.groupby("user_id", as_index=False).agg(
        avg_sentiment=("sentiment_score", "mean"),
        msg_count=("message", "size"),
        neg_ratio=("sentiment_label", lambda x: float((x == "negative").mean())),
    )
    per_user["user"] = per_user["user_id"].map(name_map)
    per_user["user_short"] = per_user["user"].apply(lambda s: shorten_user_label(s, 26))

    top_n = st.slider("Users shown in charts", min_value=8, max_value=30, value=15, step=1)
    top_users = per_user.sort_values("msg_count", ascending=False).head(top_n).copy()
    top_user_ids = top_users["user_id"].tolist()

    left, right = st.columns(2)
    with left:
        fig_avg = px.bar(
            top_users.sort_values("avg_sentiment", ascending=True),
            x="avg_sentiment",
            y="user_short",
            orientation="h",
            hover_data=["msg_count", "neg_ratio"],
            color="avg_sentiment",
            color_continuous_scale=SENTIMENT_DIVERGING_SCALE,
            color_continuous_midpoint=0.0,
            title=f"Average Sentiment Per User (Top {top_n} by Message Volume)",
        )
        style_chart(fig_avg, height=520, x_title="Average Sentiment", y_title="User")
        st.plotly_chart(fig_avg, width="stretch")

    with right:
        sent_mix = wa.groupby(["user_id", "sentiment_label"], as_index=False).size()
        sent_mix["user"] = sent_mix["user_id"].map(name_map)
        sent_mix["user_short"] = sent_mix["user"].apply(lambda s: shorten_user_label(s, 26))
        sent_mix = sent_mix[sent_mix["user_id"].isin(top_user_ids)].copy()

        totals = sent_mix.groupby("user_id", as_index=False)["size"].sum().rename(columns={"size": "total"})
        sent_mix = sent_mix.merge(totals, on="user_id", how="left")
        sent_mix["pct"] = (sent_mix["size"] / sent_mix["total"]).fillna(0.0)
        fig_mix = px.bar(
            sent_mix.sort_values("pct", ascending=False),
            x="pct",
            y="user_short",
            orientation="h",
            color="sentiment_label",
            color_discrete_map=SENTIMENT_COLORS,
            title=f"Sentiment Mix Per User (Top {top_n}, Normalized %)",
        )
        fig_mix.update_layout(barmode="stack")
        fig_mix.update_xaxes(tickformat=".0%")
        style_chart(fig_mix, height=520, x_title="Share of Messages", y_title="User")
        st.plotly_chart(fig_mix, width="stretch")

    st.markdown("### User Sentiment Leaderboard")
    MIN_LEADERBOARD_THRESHOLD = 10
    lleft, lright = st.columns(2)
    with lleft:
        st.markdown("#### Top 10 Positive Users")
        top_pos = per_user[per_user["msg_count"] >= MIN_LEADERBOARD_THRESHOLD].sort_values("avg_sentiment", ascending=False).head(10).copy()
        if top_pos.empty:
            st.info("No positive users found with sufficient volume.")
        else:
            top_pos["display_sentiment"] = top_pos["avg_sentiment"].map(lambda v: f"{v:+.3f}")
            st.dataframe(
                top_pos[["user", "msg_count", "display_sentiment"]].rename(columns={"display_sentiment": "Avg Sentiment", "msg_count": "Messages"}),
                width="stretch",
                height=300,
                hide_index=True
            )

    with lright:
        st.markdown("#### Top 10 Negative Users")
        top_neg = per_user[per_user["msg_count"] >= MIN_LEADERBOARD_THRESHOLD].sort_values("avg_sentiment", ascending=True).head(10).copy()
        if top_neg.empty:
            st.info("No negative users found with sufficient volume.")
        else:
            top_neg["display_sentiment"] = top_neg["avg_sentiment"].map(lambda v: f"{v:+.3f}")
            st.dataframe(
                top_neg[["user", "msg_count", "display_sentiment"]].rename(columns={"display_sentiment": "Avg Sentiment", "msg_count": "Messages"}),
                width="stretch",
                height=300,
                hide_index=True
            )

    st.subheader("Per-User Sentiment Trend")
    users = sorted(wa["user_id"].unique().tolist())
    selected_uid = st.selectbox("Select User", users, format_func=lambda uid: name_map.get(int(uid), f"User ({uid})"))
    u = wa[wa["user_id"] == int(selected_uid)].copy().sort_values("created_at")
    u["sentiment_score"] = pd.to_numeric(u.get("sentiment_score"), errors="coerce").fillna(0.0)
    u["created_at"] = pd.to_datetime(u.get("created_at"), errors="coerce", utc=True)
    if "sentiment_confidence" in u.columns:
        u["sentiment_confidence"] = pd.to_numeric(u.get("sentiment_confidence"), errors="coerce").fillna(0.0).clip(0.0, 1.0)
    else:
        u["sentiment_confidence"] = u["sentiment_score"].abs().clip(0.0, 1.0)

    tleft, tright = st.columns(2)
    with tleft:
        if u["created_at"].notna().any():
            t = u.dropna(subset=["created_at"]).copy()
            t["rolling_ema"] = t["sentiment_score"].ewm(span=12, adjust=False).mean()
            t["rolling_mean"] = t["sentiment_score"].rolling(window=10, min_periods=1).mean()
            fig_t = go.Figure()
            fig_t.add_trace(
                go.Scatter(
                    x=t["created_at"],
                    y=t["sentiment_score"],
                    mode="markers",
                    name="Message Score",
                    marker=dict(
                        size=(6 + 10 * t["sentiment_confidence"]).clip(5, 16),
                        color=t["sentiment_score"],
                        colorscale="RdYlGn",
                        cmin=-1,
                        cmax=1,
                        line=dict(width=0.4, color="#3a2d1f"),
                        opacity=0.62,
                    ),
                )
            )
            fig_t.add_trace(
                go.Scatter(x=t["created_at"], y=t["rolling_mean"], mode="lines", name="Rolling Mean (10)", line=dict(width=2.6, color="#1E3A5F"))
            )
            fig_t.add_trace(
                go.Scatter(x=t["created_at"], y=t["rolling_ema"], mode="lines", name="EMA (12)", line=dict(width=2.2, dash="dot", color="#8D6A3B"))
            )
            fig_t.update_layout(title="Per-Message Sentiment with Trend Smoothing")
            fig_t.add_hline(y=0.1, line_dash="dash", line_color="#2E8B57")
            fig_t.add_hline(y=-0.1, line_dash="dash", line_color="#B2413E")
            style_chart(fig_t, height=420, x_title="Timestamp", y_title="Sentiment Score")
            st.plotly_chart(fig_t, width="stretch")
        else:
            st.info("No timestamped messages for selected user.")

    with tright:
        if u["created_at"].notna().any():
            m = u.dropna(subset=["created_at"]).copy()
            m["day"] = m["created_at"].dt.floor("D")
            day_series = (
                m.groupby("day", as_index=False)
                .agg(avg_sentiment=("sentiment_score", "mean"), messages=("sentiment_score", "size"))
                .sort_values("day")
            )
            day_series["neg_share"] = (
                m.groupby("day")["sentiment_label"].apply(lambda x: float((x == "negative").mean())).reindex(day_series["day"]).values
            )
            fig_day = px.bar(
                day_series,
                x="day",
                y="avg_sentiment",
                color="neg_share",
                color_continuous_scale=[[0.0, "#79AF8E"], [1.0, "#B2413E"]],
                title="Daily Average Sentiment (Color = Negative Share)",
                hover_data=["messages", "neg_share"],
            )
            fig_day.add_hline(y=0.0, line_dash="dot", line_color="#6F5A40")
            style_chart(fig_day, height=420, x_title="Day", y_title="Average Sentiment")
            st.plotly_chart(fig_day, width="stretch")
        else:
            fig_h = px.histogram(
                u,
                x="sentiment_score",
                nbins=22,
                color="sentiment_label",
                color_discrete_map=SENTIMENT_COLORS,
                title="Sentiment Score Distribution",
            )
            style_chart(fig_h, height=420, x_title="Sentiment Score", y_title="Message Count")
            st.plotly_chart(fig_h, width="stretch")

    st.dataframe(
        u[["created_at", "message", "sentiment_score", "sentiment_label"]]
        .sort_values("created_at", ascending=False)
        .head(40),
        width="stretch",
        height=360,
    )
    return
