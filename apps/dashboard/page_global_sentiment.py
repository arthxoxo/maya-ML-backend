"""
Global Sentiment Analysis page for the Maya dashboard.
"""

from __future__ import annotations

from typing import Dict
import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

from apps.dashboard.shared import (
    SENTIMENT_COLORS, SENTIMENT_DIVERGING_SCALE, ACCENT_PRIMARY, RISK_COLORS,
    style_chart, executive_card, executive_metric,
)
from apps.dashboard.data_loaders import (
    load_user_directory,
    load_user_dissatisfaction_flags,
    load_gru_mood_swing_summary,
    load_gru_mood_training_report,
    run_gru_mood_training_action,
)


def render(wa: pd.DataFrame, user_directory: pd.DataFrame, name_map: Dict[int, str], refresh_nonce: str) -> None:
    """Render the Global Sentiment Analysis page."""
    st.subheader("Sentiment Quality Monitor")
    st.caption("Tracks sentiment confidence and label drift so weak or uncertain scoring is visible in the UI.")


    wa_quality = wa.copy()
    wa_quality["sentiment_score"] = pd.to_numeric(wa_quality.get("sentiment_score"), errors="coerce").fillna(0.0).clip(-1.0, 1.0)
    if "sentiment_confidence" in wa_quality.columns:
        conf_raw = pd.to_numeric(wa_quality.get("sentiment_confidence"), errors="coerce")
        if conf_raw.isna().all() or float(conf_raw.fillna(0.0).max()) <= 0.01:
            wa_quality["confidence_proxy"] = wa_quality["sentiment_score"].abs().clip(0.0, 1.0)
        else:
            wa_quality["confidence_proxy"] = conf_raw.fillna(wa_quality["sentiment_score"].abs()).clip(0.0, 1.0)
    else:
        wa_quality["confidence_proxy"] = wa_quality["sentiment_score"].abs().clip(0.0, 1.0)
    wa_quality["sentiment_label"] = (
        wa_quality.get("sentiment_label", "")
        .fillna("")
        .astype(str)
        .str.lower()
        .str.strip()
        .replace("", "neutral")
    )
    wa_quality["created_at"] = pd.to_datetime(wa_quality.get("created_at"), errors="coerce", utc=True)

    latest_ts = wa_quality["created_at"].max() if wa_quality["created_at"].notna().any() else pd.NaT
    if pd.notna(latest_ts):
        recent_cutoff = latest_ts - pd.Timedelta(days=14)
        prev_cutoff = recent_cutoff - pd.Timedelta(days=14)
        recent = wa_quality[wa_quality["created_at"] >= recent_cutoff].copy()
        previous = wa_quality[(wa_quality["created_at"] >= prev_cutoff) & (wa_quality["created_at"] < recent_cutoff)].copy()
    else:
        recent = wa_quality.copy()
        previous = pd.DataFrame(columns=wa_quality.columns)

    recent_conf = float(recent["confidence_proxy"].mean()) if not recent.empty else float(wa_quality["confidence_proxy"].mean())
    prev_conf = float(previous["confidence_proxy"].mean()) if not previous.empty else np.nan
    low_conf_rate = float((wa_quality["confidence_proxy"] < 0.20).mean()) if not wa_quality.empty else 0.0
    recent_neg_ratio = float((recent["sentiment_label"] == "negative").mean()) if not recent.empty else float((wa_quality["sentiment_label"] == "negative").mean())
    prev_neg_ratio = float((previous["sentiment_label"] == "negative").mean()) if not previous.empty else np.nan
    neg_drift = recent_neg_ratio - prev_neg_ratio if pd.notna(prev_neg_ratio) else np.nan
    conf_drift = recent_conf - prev_conf if pd.notna(prev_conf) else np.nan

    q1, q2, q3, q4 = st.columns(4)
    with q1: executive_metric("Avg Confidence (14d)", f"{recent_conf:.1%}", delta=f"{conf_drift:+.1%}")
    with q2: executive_metric("Low-Confidence Rate", f"{low_conf_rate:.1%}")
    with q3: executive_metric("Negative Share (14d)", f"{recent_neg_ratio:.1%}", delta=f"{neg_drift:+.1%}")
    with q4: executive_metric("Uncertain Messages", f"{int((wa_quality['confidence_proxy'] < 0.20).sum()):,}")

    gleft, gright = st.columns(2)
    with gleft:
        dist_df = wa_quality.copy()
        dist_df["sentiment_label"] = dist_df["sentiment_label"].where(
            dist_df["sentiment_label"].isin(["positive", "negative", "neutral"]),
            "neutral",
        )
        fig_dist = px.histogram(
            dist_df,
            x="sentiment_score",
            color="sentiment_label",
            color_discrete_map=SENTIMENT_COLORS,
            nbins=50,
            barmode="overlay",
            opacity=0.58,
            title="Sentiment Score Distribution by Label",
        )
        fig_dist.add_vline(x=0.0, line_dash="dot", line_color="#6F5A40")
        style_chart(fig_dist, height=410, x_title="Sentiment Score", y_title="Messages")
        st.plotly_chart(fig_dist, width="stretch")

    with gright:
        temporal = wa_quality.dropna(subset=["created_at"]).copy()
        if temporal.empty:
            st.info("No timestamps available for temporal sentiment diagnostics.")
        else:
            temporal["hour"] = temporal["created_at"].dt.hour
            temporal["day_name"] = temporal["created_at"].dt.day_name()
            day_order = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
            temporal["day_name"] = pd.Categorical(temporal["day_name"], categories=day_order, ordered=True)
            heat = (
                temporal.groupby(["day_name", "hour"], observed=True, as_index=False)
                .agg(neg_share=("sentiment_label", lambda x: float((x == "negative").mean())))
            )
            fig_heat = px.density_heatmap(
                heat,
                x="hour",
                y="day_name",
                z="neg_share",
                color_continuous_scale=[[0.0, "#e9f4ec"], [0.5, "#f1cf8a"], [1.0, "#b2413e"]],
                title="Negative Share Heatmap (Hour x Weekday)",
            )
            style_chart(fig_heat, height=410, x_title="Hour of Day", y_title="Weekday")
            fig_heat.update_xaxes(dtick=2)
            fig_heat.update_coloraxes(colorbar_title="Neg Share")
            st.plotly_chart(fig_heat, width="stretch")

    qleft, qright = st.columns(2)
    with qleft:
        conf_series = wa_quality.dropna(subset=["created_at"]).copy()
        if conf_series.empty:
            st.info("No timestamps available to compute confidence trend.")
        else:
            conf_series["day"] = conf_series["created_at"].dt.floor("D")
            conf_trend = (
                conf_series.groupby("day", as_index=False)
                .agg(confidence=("confidence_proxy", "mean"), messages=("message", "size"))
                .sort_values("day")
            )
            conf_trend["confidence_7d"] = conf_trend["confidence"].rolling(window=7, min_periods=1).mean()
            fig_conf = px.line(
                conf_trend,
                x="day",
                y=["confidence", "confidence_7d"],
                title="Sentiment Confidence Trend",
            )
            fig_conf.update_traces(mode="lines+markers")
            style_chart(fig_conf, height=420, x_title="Date", y_title="Confidence")
            fig_conf.update_yaxes(range=[0, 1], tickformat=".0%")
            st.plotly_chart(fig_conf, width="stretch")

    with qright:
        drift = wa_quality.dropna(subset=["created_at"]).copy()
        if drift.empty:
            st.info("No timestamps available to compute label drift.")
        else:
            drift["week_start"] = drift["created_at"].dt.to_period("W").dt.start_time
            drift_mix = drift.groupby(["week_start", "sentiment_label"], as_index=False).size()
            totals = drift_mix.groupby("week_start", as_index=False)["size"].sum().rename(columns={"size": "total"})
            drift_mix = drift_mix.merge(totals, on="week_start", how="left")
            drift_mix["share"] = (drift_mix["size"] / drift_mix["total"]).fillna(0.0)
            drift_mix = drift_mix.sort_values("week_start")
            fig_drift = px.bar(
                drift_mix,
                x="week_start",
                y="share",
                color="sentiment_label",
                color_discrete_map=SENTIMENT_COLORS,
                title="Weekly Sentiment Label Drift",
            )
            fig_drift.update_layout(barmode="stack")
            style_chart(fig_drift, height=420, x_title="Week", y_title="Share", rotate_x=True)
            fig_drift.update_yaxes(range=[0, 1], tickformat=".0%")
            fig_drift.update_xaxes(
                tickformat="%Y-%m-%d",
                tickangle=-35,
                nticks=8,
            )
            fig_drift.update_layout(
                margin=dict(l=62, r=22, t=92, b=92),
                legend=dict(
                    orientation="h",
                    yanchor="bottom",
                    y=1.02,
                    xanchor="right",
                    x=1.0,
                    title_text="",
                    font=dict(size=12, color="#1e2228"),
                    bgcolor="rgba(255,251,243,0.96)",
                ),
            )
            st.plotly_chart(fig_drift, width="stretch")

    st.divider()
    st.subheader("Global Dissatisfaction Analysis")
    dissatisfaction_df = load_user_dissatisfaction_flags()
    if dissatisfaction_df.empty:
        st.info("Dissatisfaction scores unavailable (sentiment_scores.csv not found).")
    else:
        avg_risk = dissatisfaction_df["dissatisfaction_score"].mean()
        high_risk_count = (dissatisfaction_df["dissatisfaction_flag"] == "High").sum()
        med_risk_count = (dissatisfaction_df["dissatisfaction_flag"] == "Medium").sum()

        r1, r2, r3 = st.columns(3)
        r1.metric("Population Avg Risk", f"{avg_risk:.3f}")
        r1.caption("Composite score of negative ratio and sentiment strength.")

        r2.metric("High Risk Users", f"{high_risk_count:,}")
        r2.caption(f"{(high_risk_count/len(dissatisfaction_df)):.1%} of tracked population.")

        r3.metric("Medium Risk Users", f"{med_risk_count:,}")
        r3.caption(f"{(med_risk_count/len(dissatisfaction_df)):.1%} of tracked population.")

        risk_counts = dissatisfaction_df.groupby("dissatisfaction_flag").size().reset_index(name="users")
        order = ["High", "Medium", "Low"]
        risk_counts["dissatisfaction_flag"] = pd.Categorical(risk_counts["dissatisfaction_flag"], categories=order, ordered=True)
        risk_counts = risk_counts.sort_values("dissatisfaction_flag")
        fig_risk = px.bar(
            risk_counts,
            x="dissatisfaction_flag",
            y="users",
            color="dissatisfaction_flag",
            color_discrete_map=RISK_COLORS,
            title="User Distribution By Dissatisfaction Risk",
        )
        style_chart(fig_risk, height=450, x_title="Risk Level", y_title="Users")
        st.plotly_chart(fig_risk, width="stretch")
        st.caption("Risk categories are derived from the population percentiles of the dissatisfaction scores.")

        st.subheader("Most Vulnerable Users")
        # Only include users with enough messages for a meaningful signal
        MIN_MSG_THRESHOLD = 10
        qualified = dissatisfaction_df[dissatisfaction_df["msg_count"] >= MIN_MSG_THRESHOLD].copy()
        if qualified.empty:
            st.info(f"No users with ≥{MIN_MSG_THRESHOLD} messages to rank.")
        else:
            top_vulnerable = qualified.sort_values("dissatisfaction_score", ascending=False).head(15).copy()
            if not user_directory.empty:
                names = user_directory[["user_id", "display_name"]].drop_duplicates("user_id")
                top_vulnerable = top_vulnerable.merge(names, on="user_id", how="left")
                top_vulnerable["user_label"] = top_vulnerable["display_name"].fillna("User " + top_vulnerable["user_id"].astype(str))
            else:
                top_vulnerable["user_label"] = "User " + top_vulnerable["user_id"].astype(str)

            fig_vulnerable = px.bar(
                top_vulnerable.sort_values("dissatisfaction_score", ascending=True),
                x="dissatisfaction_score",
                y="user_label",
                orientation="h",
                title="Top 15 Most Vulnerable Users",
                hover_data=["dissatisfaction_reason", "avg_sentiment", "neg_ratio", "msg_count"]
            )
            style_chart(fig_vulnerable, height=550, x_title="Dissatisfaction Score", y_title="User")
            fig_vulnerable.update_traces(marker_color="#B2413E")
            st.plotly_chart(fig_vulnerable, width="stretch")
            st.caption(f"Showing users with ≥{MIN_MSG_THRESHOLD} messages. Hover for risk drivers.")

    return
