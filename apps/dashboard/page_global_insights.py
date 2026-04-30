"""
Global Insights page for the Maya dashboard.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

from apps.dashboard.shared import (
    ACCENT_PRIMARY, SENTIMENT_COLORS,
    style_chart,
)
from apps.dashboard.data_loaders import (
    build_task_importance,
    build_feature_focus_summary,
    build_representative_statements,
)


def render(scores: pd.DataFrame, sentiment_df: pd.DataFrame) -> None:
    """Render the Global Insights page."""
    if page == "Global Insights":
        global_tasks = build_task_importance(sentiment_df, user_id=None, top_k=20).fillna(0)
        global_feature_focus = build_feature_focus_summary(sentiment_df, top_k=12).fillna(0)
        if "sample_requests" in global_feature_focus.columns:
            global_feature_focus["sample_requests"] = global_feature_focus["sample_requests"].fillna("")
        
        global_statements = build_representative_statements(sentiment_df, user_id=None, top_k=12).fillna(0)

        g1, g2, g3 = st.columns(3)
        g1.metric("Total Users", f"{scores['user_id'].nunique()}")
        g2.metric("Users With Chat Sentiment", f"{sentiment_df['user_id'].nunique() if not sentiment_df.empty else 0}")
        g3.metric("Global Avg Sentiment", f"{sentiment_df['polarity'].mean():.3f}" if not sentiment_df.empty else "N/A")

        left, right = st.columns(2)
        with left:
            st.subheader("Global Sentiment Distribution")
            if sentiment_df.empty:
                st.info("No sentiment rows found in preprocessed messages.")
            else:
                sent_counts = sentiment_df.copy()
                sent_counts["sentiment"] = sent_counts["sentiment"].fillna("neutral").astype(str).str.lower().str.strip()
                sent_counts = sent_counts[sent_counts["sentiment"].isin(["positive", "negative", "neutral"])]
                sent_counts = sent_counts.groupby("sentiment").size().reset_index(name="count")
                fig_pie = px.pie(
                    sent_counts,
                    names="sentiment",
                    values="count",
                    hole=0.45,
                    color="sentiment",
                    color_discrete_map=SENTIMENT_COLORS,
                    category_orders={"sentiment": ["positive", "neutral", "negative"]},
                )
                fig_pie = style_chart(fig_pie, height=420, kind="pie")
                st.plotly_chart(fig_pie, use_container_width=True)

        with right:
            st.subheader("Global Sentiment Over Time")
            time_df = sentiment_df.dropna(subset=["created_at"]).copy()
            if time_df.empty:
                st.info("No timestamped sentiment rows available.")
            else:
                time_df["date"] = time_df["created_at"].dt.date
                daily = time_df.groupby("date", as_index=False).agg(avg_polarity=("polarity", "mean")).fillna(0)
                daily["date_str"] = daily["date"].astype(str)
                fig_time = px.line(daily, x="date_str", y="avg_polarity", markers=True)
                fig_time.add_hline(y=0.1, line_dash="dash", line_color="rgba(255,255,255,0.2)")
                fig_time.add_hline(y=-0.1, line_dash="dash", line_color="rgba(255,255,255,0.2)")
                fig_time = style_chart(fig_time, height=420, x_title="Date", y_title="Average Polarity")
                st.plotly_chart(fig_time, use_container_width=True)

        st.subheader("RAG Focus Opportunities (Global User Requests)")
        if global_feature_focus.empty:
            st.info("No clustered feature-demand signals found yet.")
        else:
            fig_focus = px.bar(
                global_feature_focus.sort_values("mentions", ascending=True),
                x="mentions",
                y="feature_focus",
                orientation="h",
                color="share",
                color_continuous_scale=[[0.0, "#dce9f5"], [1.0, ACCENT_PRIMARY]],
                title="Most Requested Capability Clusters",
                hover_data=["share", "avg_polarity", "sample_requests"],
            )
            style_chart(fig_focus, height=440, x_title="Mentions", y_title="Capability Cluster")
            st.plotly_chart(fig_focus, width="stretch")
            focus_table = global_feature_focus.copy()
            focus_table["share"] = focus_table["share"].map(lambda v: f"{v:.1%}")
            focus_table["avg_polarity"] = focus_table["avg_polarity"].map(lambda v: f"{v:.3f}")
