"""
RAG Roadmap Signals page for the Maya dashboard.
"""

from __future__ import annotations

from typing import Dict
import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

from apps.dashboard.shared import (
    ACCENT_PRIMARY, SENTIMENT_COLORS, RISK_COLORS,
    style_chart, executive_metric,
)
from apps.dashboard.data_loaders import (
    load_sentiment_table,
    load_user_directory,
    load_xgb_user_predictions,
    compute_xgb_prediction_health,
    build_rag_roadmap_signals,
    build_tool_usage_signals,
)


def render(refresh_nonce: str) -> None:
    """Render the RAG Roadmap Signals page."""
    if True:
        sentiment_df = load_sentiment_table(refresh_nonce)
        roadmap = build_rag_roadmap_signals(sentiment_df, recent_days=30, top_k=14)
        st.subheader("RAG Roadmap Signals")
        st.caption("Prioritize capabilities with strong demand, rising trend, and higher dissatisfaction.")

        # Embedding Model Predictions — shown at the top of the page
        pred_df = load_xgb_user_predictions()

        if not pred_df.empty:
            names = load_user_directory()
            pred_view = pred_df.copy()
            if not names.empty:
                pred_view = pred_view.merge(names, on="user_id", how="left")
                pred_view["user"] = pred_view["display_name"].fillna("User (" + pred_view["user_id"].astype(str) + ")")
            else:
                pred_view["user"] = "User (" + pred_view["user_id"].astype(str) + ")"
            pred_view["pred_prob_negative"] = 1.0 - pred_view["pred_prob_positive"].astype(float)
            pred_view = pred_view.sort_values("pred_prob_positive", ascending=False)

            # Exclude insufficient_data users — they have no real signal
            valid_pred = pred_view[pred_view["predicted_class"].astype(str).str.lower().str.strip() != "insufficient_data"].copy()
            pos_users = valid_pred[valid_pred["pred_prob_positive"] >= 0.5]
            neg_users = valid_pred[valid_pred["pred_prob_positive"] < 0.5]
            pos_pct = f"{len(pos_users) / len(valid_pred):.1%}" if len(valid_pred) > 0 else "0%"
            neg_pct = f"{len(neg_users) / len(valid_pred):.1%}" if len(valid_pred) > 0 else "0%"
            avg_pos_conf = f"{pos_users['confidence'].mean():.1%}" if not pos_users.empty else "N/A"
            avg_neg_conf = f"{neg_users['confidence'].mean():.1%}" if not neg_users.empty else "N/A"

            card_left, card_right = st.columns(2)
            with card_left:
                st.markdown(f"""
                <div style="background: linear-gradient(135deg, #1a3a2a 0%, #0d1117 100%); border: 1px solid #2E8B57; border-radius: 12px; padding: 28px 24px; text-align: center;">
                    <div style="font-size: 14px; color: #8b9dc3; text-transform: uppercase; letter-spacing: 1.5px; margin-bottom: 8px;">Positive Predictions</div>
                    <div style="font-size: 42px; font-weight: 700; color: #2E8B57; margin-bottom: 4px;">{len(pos_users):,}</div>
                    <div style="font-size: 14px; color: #a0aec0; margin-bottom: 16px;">{pos_pct} of all users</div>
                    <div style="border-top: 1px solid rgba(46,139,87,0.3); padding-top: 14px; font-size: 13px; color: #8b9dc3;">
                        Avg Confidence: <span style="color: #2E8B57; font-weight: 600;">{avg_pos_conf}</span>
                    </div>
                </div>
                """, unsafe_allow_html=True)
            with card_right:
                st.markdown(f"""
                <div style="background: linear-gradient(135deg, #3a1a1a 0%, #0d1117 100%); border: 1px solid #B2413E; border-radius: 12px; padding: 28px 24px; text-align: center;">
                    <div style="font-size: 14px; color: #8b9dc3; text-transform: uppercase; letter-spacing: 1.5px; margin-bottom: 8px;">Negative Predictions</div>
                    <div style="font-size: 42px; font-weight: 700; color: #B2413E; margin-bottom: 4px;">{len(neg_users):,}</div>
                    <div style="font-size: 14px; color: #a0aec0; margin-bottom: 16px;">{neg_pct} of all users</div>
                    <div style="border-top: 1px solid rgba(178,65,62,0.3); padding-top: 14px; font-size: 13px; color: #8b9dc3;">
                        Avg Confidence: <span style="color: #B2413E; font-weight: 600;">{avg_neg_conf}</span>
                    </div>
                </div>
                """, unsafe_allow_html=True)

        st.divider()

        if roadmap.empty:
            st.info("No canonical intent requests found yet to build roadmap signals.")
        else:
            r1, r2, r3 = st.columns(3)
            r1.metric("Tracked Intents", f"{roadmap['intent'].nunique()}")
            r2.metric("Top Opportunity", str(roadmap.iloc[0]['intent']))
            r3.metric("Top Score", f"{float(roadmap.iloc[0]['opportunity_score']):.1f}")

            fig_opp = px.bar(
                roadmap.sort_values("opportunity_score", ascending=True),
                x="opportunity_score",
                y="intent",
                orientation="h",
                color="neg_ratio",
                color_continuous_scale=[[0.0, "#d8ead9"], [1.0, "#b2413e"]],
                hover_data=["mentions", "recent_mentions", "previous_mentions", "trend_pct", "avg_polarity"],
                title="Intent Opportunity Ranking",
            )
            style_chart(fig_opp, height=560, x_title="Opportunity Score", y_title="Intent")
            st.plotly_chart(fig_opp, width="stretch")

            roadmap_view = roadmap.copy()
            roadmap_view["share"] = roadmap_view["share"].map(lambda v: f"{v:.1%}")
            roadmap_view["neg_ratio"] = roadmap_view["neg_ratio"].map(lambda v: f"{v:.1%}")
            roadmap_view["trend_pct"] = roadmap_view["trend_pct"].map(lambda v: f"{v:+.0%}")
            roadmap_view["avg_polarity"] = roadmap_view["avg_polarity"].map(lambda v: f"{v:.3f}")
            st.dataframe(roadmap_view, width="stretch", height=390)

        st.divider()

        # Humanoid Bot Tool Usage Section
        st.markdown("### Humanoid Bot Tool Usage")
        st.caption("Tracking specific tool mentions (Calendar, Reminders, Notes, etc.) from user requests.")
        
        tool_signals = build_tool_usage_signals(sentiment_df)
        total_m = tool_signals["mentions"].sum()
        
        if total_m == 0:
            st.info("No explicit tool usage detected in recent messages yet. The data table below will populate once usage occurs.")
        else:
            cat_summary = tool_signals.groupby("category", as_index=False).agg(
                total_mentions=("mentions", "sum")
            ).sort_values("total_mentions", ascending=False)
            
            t1, t2 = st.columns(2)
            with t1:
                top_cat = cat_summary.iloc[0]["category"] if not cat_summary.empty else "N/A"
                st.metric("Most Requested Toolset", top_cat)
            with t2:
                least_cat = cat_summary.iloc[-1]["category"] if not cat_summary.empty else "N/A"
                st.metric("Least Requested Toolset", least_cat)

            fig_tools = px.treemap(
                tool_signals[tool_signals["mentions"] > 0],
                path=["category", "tool_action"],
                values="mentions",
                color="avg_polarity",
                color_continuous_scale=[[0.0, "#b2413e"], [0.5, "#d8ead9"], [1.0, "#2e8b57"]],
                color_continuous_midpoint=0.0,
                title="Tool Mentions Heatmap (Size = Volume, Color = Sentiment)"
            )
            fig_tools.update_layout(margin=dict(t=50, l=10, r=10, b=10), paper_bgcolor="rgba(0,0,0,0)")
            st.plotly_chart(fig_tools, use_container_width=True)

        st.markdown("#### Detailed Tool Breakdown")
        tool_view = tool_signals.copy()
        tool_view.rename(columns={
            "category": "Category",
            "tool_action": "Tool Action",
            "mentions": "Mentions",
            "avg_polarity": "Avg Polarity",
            "neg_ratio": "Negative Ratio",
            "positive_sample": "Positive Example",
            "negative_sample": "Negative Example"
        }, inplace=True)
        tool_view["Avg Polarity"] = tool_view["Avg Polarity"].map(lambda v: f"{v:.3f}")
        tool_view["Negative Ratio"] = tool_view["Negative Ratio"].map(lambda v: f"{v:.1%}")
        
        st.dataframe(tool_view, width="stretch", height=300)

        st.divider()

        pred_summary = load_xgb_user_predictions()
        if not pred_summary.empty:
            pred_summary = pred_summary.copy()
            pred_summary["pred_prob_positive"] = pd.to_numeric(pred_summary.get("pred_prob_positive"), errors="coerce").fillna(0.0)
            pred_summary["predicted_class"] = pred_summary.get("predicted_class", "").fillna("").astype(str).str.lower().str.strip()
            missing_class = pred_summary["predicted_class"].eq("")
            if missing_class.any():
                pred_summary.loc[missing_class, "predicted_class"] = np.where(
                    pred_summary.loc[missing_class, "pred_prob_positive"] >= 0.5,
                    "positive",
                    "negative",
                )

            total_users = int(pred_summary["user_id"].nunique()) if "user_id" in pred_summary.columns else int(len(pred_summary))
            positive_count = int((pred_summary["predicted_class"] == "positive").sum())
            negative_count = int((pred_summary["predicted_class"] == "negative").sum())
            health = compute_xgb_prediction_health(pred_summary)

            st.markdown("### Prediction Snapshot")
            s1, s2, s3, s4 = st.columns(4)
            with s1: executive_metric("Total Users", f"{total_users:,}")
            with s2: executive_metric("Positive Forecasts", f"{positive_count:,}")
            with s3: executive_metric("Negative Forecasts", f"{negative_count:,}")
            with s4: executive_metric("Class Concentration", f"{float(health['dominance']):.1%}")

            # Confidence Gate tier metrics
            gate_col = pred_summary.get("confidence_gate", pd.Series(dtype=str)).fillna("manual_review").astype(str).str.strip().str.lower()
            gate_auto = int((gate_col == "auto").sum())
            gate_review = int((gate_col == "review").sum())
            gate_manual = int((gate_col == "manual_review").sum())
            g1, g2, g3 = st.columns(3)
            with g1: executive_metric("✅ Auto (≥70%)", f"{gate_auto:,}")
            with g2: executive_metric("🔍 Review (30–70%)", f"{gate_review:,}")
            with g3: executive_metric("⚠️ Manual Review (<30%)", f"{gate_manual:,}")

            if bool(health["collapse_flag"]):
                st.error(
                    "Model collapse risk detected: predictions are dominated by one class with very low variation. "
                    "Recheck target balance and training labels before trusting SHAP outputs."
                )
            elif float(health["dominance"]) >= 0.85:
                if float(health.get("probability_std", np.nan)) <= 0.10:
                    st.warning("Prediction mix is heavily skewed. Monitor class balance and probability spread each run.")

            improve_view = pred_summary.copy()
            improve_view["confidence"] = pd.to_numeric(improve_view.get("confidence"), errors="coerce")
            if improve_view["confidence"].isna().all():
                improve_view["confidence"] = (2.0 * (improve_view["pred_prob_positive"] - 0.5).abs()).clip(0.0, 1.0)
            else:
                improve_view["confidence"] = improve_view["confidence"].fillna(
                    (2.0 * (improve_view["pred_prob_positive"] - 0.5).abs()).clip(0.0, 1.0)
                )
            # Ensure confidence_gate column exists
            if "confidence_gate" not in improve_view.columns:
                conf = improve_view["confidence"]
                improve_view["confidence_gate"] = np.where(conf >= 0.70, "auto", np.where(conf >= 0.30, "review", "manual_review"))

            names = load_user_directory()
            if not names.empty and "user_id" in improve_view.columns:
                improve_view = improve_view.merge(names, on="user_id", how="left")
                improve_view["user"] = improve_view["display_name"].fillna("User (" + improve_view["user_id"].astype(str) + ")")
            elif "user_id" in improve_view.columns:
                improve_view["user"] = "User (" + improve_view["user_id"].astype(str) + ")"
            else:
                improve_view["user"] = "User"

            def recommendation_for_row(row: pd.Series) -> str:
                prob_pos = float(row.get("pred_prob_positive", 0.0))
                confidence = float(row.get("confidence", 0.0))
                pred_class = str(row.get("predicted_class", "")).strip().lower()
                gate = str(row.get("confidence_gate", "manual_review")).strip().lower()

                if gate == "manual_review":
                    return "⚠️ Manual Review Required — model confidence too low for automated advice."
                if gate == "review":
                    return "🔍 Needs Review — borderline confidence; verify with recent user data before acting."
                # Auto tier: generate specific improvement tips
                if 0.45 <= prob_pos <= 0.55:
                    return "Borderline score; add more labeled feedback samples for this user."
                if pred_class == "negative" and prob_pos < 0.25:
                    return "Strong negative signal; review message context and add nuanced sentiment labels."
                if pred_class == "positive" and prob_pos > 0.80:
                    return "✅ Stable prediction; keep monitoring for drift with periodic label audits."
                return "Improve feature coverage using richer behavioral and interaction-level signals."

            improve_view["improvement_action"] = improve_view.apply(recommendation_for_row, axis=1)

            # Tier filter
            gate_filter = st.selectbox(
                "Filter By Confidence Tier",
                ["All", "✅ Auto (High Confidence)", "🔍 Review (Borderline)", "⚠️ Manual Review (Low Confidence)"],
                key="gate_filter",
            )
            filtered_view = improve_view.copy()
            if "Auto" in gate_filter:
                filtered_view = filtered_view[filtered_view["confidence_gate"] == "auto"]
            elif "Review" in gate_filter and "Manual" not in gate_filter:
                filtered_view = filtered_view[filtered_view["confidence_gate"] == "review"]
            elif "Manual" in gate_filter:
                filtered_view = filtered_view[filtered_view["confidence_gate"] == "manual_review"]

            show_cols = [
                c
                for c in ["user", "user_id", "predicted_class", "pred_prob_positive", "confidence", "confidence_gate", "improvement_action"]
                if c in filtered_view.columns
            ]
            st.dataframe(
                filtered_view[show_cols].sort_values(["confidence", "pred_prob_positive"], ascending=[True, True]),
                width="stretch",
                height=330,
            )
        else:
            st.info("Prediction snapshot card is unavailable because per-user XGBoost predictions were not found.")
        return
