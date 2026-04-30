from __future__ import annotations

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

from apps.dashboard.shared import (
    style_chart, remove_geographic_noise, remove_non_actionable_feature_noise,
    SENTIMENT_COLORS
)
from apps.dashboard.data_loaders import (
    load_sentiment_table, load_user_message_events, build_user_snapshot,
    build_representative_statements, load_user_profiles, load_persona_outputs,
    _derive_city_state, build_latest_interaction_scores
)

def render(refresh_nonce: str, display_map: dict[int, str], score_users: list[int], scores: pd.DataFrame, per_user_imp: pd.DataFrame, sentiment_df: pd.DataFrame) -> None:
    st.subheader("Per-User Controls")
    control_left, control_mid, control_right = st.columns([2.5, 2.2, 1.0])
    
    sentiment_user_ids = set(sentiment_df["user_id"].dropna().astype(int).unique()) if not sentiment_df.empty else set()
    active_users = sorted([u for u in score_users if u in sentiment_user_ids]) if sentiment_user_ids else score_users

    with control_left:
        search_query = st.text_input(
            "Search User (Name or ID)",
            value=st.session_state.get("per_user_search_query", ""),
            placeholder="Try: tushi, joshi, or 123",
            key="per_user_search_query",
        ).strip().lower()
    
    matching_users = [
        uid for uid in active_users
        if not search_query or search_query in display_map.get(uid, "").lower() or search_query in str(uid)
    ]
    if not matching_users:
        st.warning("No users matched your search. Showing full user list.")
        matching_users = active_users

    current_selected = int(st.session_state.get("selected_user_per_user", matching_users[0]))
    if current_selected not in matching_users:
        current_selected = int(matching_users[0])

    with control_mid:
        selected_user = st.selectbox(
            "Select User",
            matching_users,
            index=matching_users.index(current_selected),
            format_func=lambda uid: display_map.get(uid, f"User ({uid})"),
            key="per_user_selectbox",
        )
        st.session_state["selected_user_per_user"] = int(selected_user)

    if "refresh_by_user" not in st.session_state:
        st.session_state["refresh_by_user"] = {}
    refresh_by_user = st.session_state["refresh_by_user"]
    if selected_user not in refresh_by_user:
        refresh_by_user[selected_user] = 0

    with control_right:
        st.markdown("<div style='height: 1.9rem;'></div>", unsafe_allow_html=True)
        if st.button("Refresh", key="per_user_refresh_btn", use_container_width=True):
            refresh_by_user[selected_user] += 1
            load_sentiment_table.clear()
            load_user_message_events.clear()
            build_user_snapshot.clear()
            st.rerun()

    user_sent, task_imp = build_user_snapshot(selected_user, refresh_by_user[selected_user])
    user_statements = build_representative_statements(sentiment_df, user_id=selected_user, top_k=10)
    user_score_row = scores[scores["user_id"] == selected_user].head(1)
    user_imp = per_user_imp[per_user_imp["user_id"] == selected_user].sort_values("rank")
    user_imp = remove_geographic_noise(user_imp, feature_col="feature")
    user_imp = remove_non_actionable_feature_noise(user_imp, feature_col="feature")
    user_imp = user_imp[pd.to_numeric(user_imp["importance"], errors="coerce").fillna(0.0) > 0].copy()
    selected_name = display_map.get(selected_user, f"User ({selected_user})")

    st.markdown("### Snapshot")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("User", selected_name)
    c2.metric("Engagement Score", f"{float(user_score_row['engagement_score'].iloc[0]):.2f}" if not user_score_row.empty else "N/A")
    c3.metric("High Engagement Probability", f"{float(user_score_row['pred_high_engagement_prob'].iloc[0]):.2%}" if not user_score_row.empty else "N/A")

    avg_sent = user_sent["polarity"].mean() if not user_sent.empty else np.nan
    c4.metric("Avg Sentiment", f"{avg_sent:.3f}" if not np.isnan(avg_sent) else "N/A")

    st.divider()
    st.subheader("User Health Cards")
    user_profiles = load_user_profiles()
    persona_table, _, _ = load_persona_outputs()
    profile_row = user_profiles[user_profiles["user_id"] == selected_user].head(1)
    persona_row = persona_table[persona_table["user_id"] == selected_user].head(1) if not persona_table.empty else pd.DataFrame()
    persona_label = str(persona_row["persona_label"].iloc[0]) if not persona_row.empty and "persona_label" in persona_row.columns else "Unassigned"

    bio_col, radar_col = st.columns(2)
    with bio_col:
        st.markdown("#### User Bio Card")
        if profile_row.empty:
            st.info("No profile metadata found for this user.")
        else:
            created_at = pd.to_datetime(profile_row["created_at"].iloc[0], errors="coerce", utc=True)
            account_age = int((pd.Timestamp.now(tz="UTC") - created_at).days) if pd.notna(created_at) else None
            city, state = _derive_city_state(profile_row.iloc[0])
            b1, b2, b3 = st.columns(3)
            b1.metric("Account Age", f"{account_age} days" if account_age is not None else "Unknown")
            b2.metric("Location", f"{city}, {state}")
            b3.metric("Persona Label", persona_label)

    with radar_col:
        st.markdown("#### Sentiment Radar")
        radar_scores = build_latest_interaction_scores(user_sent, top_n=5)
        if radar_scores.empty:
            st.info("No recent interactions available for radar scoring.")
        else:
            emotion = float(radar_scores["emotion"].mean())
            intent = float(radar_scores["intent"].mean())
            aspect = float(radar_scores["aspect"].mean())
            theta = ["Emotion", "Intent", "Aspect", "Emotion"]
            r = [emotion, intent, aspect, emotion]
            
            fig_radar = go.Figure(data=go.Scatterpolar(
                r=r,
                theta=theta,
                fill='toself',
                marker_color='#d4af37',
            ))
            fig_radar.update_layout(
                polar=dict(
                    radialaxis=dict(visible=True, range=[0, 100], gridcolor="rgba(255,255,255,0.1)"),
                    bgcolor="rgba(0,0,0,0)",
                ),
                paper_bgcolor="rgba(0,0,0,0)",
                margin=dict(l=40, r=40, t=40, b=40),
            )
            st.plotly_chart(fig_radar, use_container_width=True)
            st.caption("Scores summarize the latest 5 user interactions.")

    st.divider()
    g1, g2 = st.columns(2)
    with g1:
        st.markdown("#### Embedded Feature Attributions (SHAP)")
        if user_imp.empty:
            st.info("No embedding feature importance found for this user.")
        else:
            fig_user_imp = px.bar(
                user_imp.tail(15),
                x="importance",
                y="feature",
                orientation="h",
                title=f"Top Features Driving Predictions for User {selected_user}",
            )
            style_chart(fig_user_imp, height=460, x_title="SHAP Value", y_title="GNN/Behavioral Feature")
            st.plotly_chart(fig_user_imp, width="stretch")

    with g2:
        st.markdown("#### Actionable Intent Signatures")
        if task_imp.empty:
            st.info("No explicit task intents derived from recent messages.")
        else:
            fig_chat = px.bar(
                task_imp.head(10).sort_values("mentions", ascending=True),
                x="mentions",
                y="task",
                orientation="h",
                color="avg_polarity",
                color_continuous_scale=[[0.0, "#E74C3C"], [0.5, "#94A3B8"], [1.0, "#2ECC71"]],
                color_continuous_midpoint=0.0,
                title="User Intent Importance",
            )
            style_chart(fig_chat, height=460, x_title="Importance", y_title="Task")
            st.plotly_chart(fig_chat, width="stretch")
            st.dataframe(task_imp[["task", "mentions", "avg_polarity", "sample_request"]].head(10), width="stretch", height=260)

    st.divider()
    st.subheader("Representative Statements In Context")
    if user_statements.empty:
        st.info("No contextual statements found for this user.")
    else:
        st.dataframe(user_statements, width="stretch", height=320)

    st.divider()
    st.subheader("Per-User Sentiment")
    if user_sent.empty:
        st.info("No user messages found for this user in gnn_preprocessed/messages_nodes.csv.")
    else:
        d1, d2 = st.columns(2)
        with d1:
            sent_counts = user_sent.copy()
            sent_counts["sentiment"] = sent_counts["sentiment"].fillna("").astype(str).str.lower().str.strip()
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
            style_chart(fig_pie, height=400, kind="pie")
            st.plotly_chart(fig_pie, width="stretch")

        with d2:
            time_df = user_sent.dropna(subset=["created_at"]).sort_values("created_at").copy()
            if not time_df.empty:
                time_df["rolling_avg"] = time_df["polarity"].rolling(window=min(10, len(time_df)), min_periods=1).mean()
                
                fig_time = go.Figure()
                fig_time.add_trace(go.Scatter(
                    x=time_df["created_at"],
                    y=time_df["polarity"],
                    mode="markers",
                    name="Raw Message",
                    marker=dict(color="#d4af37", size=6, opacity=0.35),
                    showlegend=False
                ))
                fig_time.add_trace(go.Scatter(
                    x=time_df["created_at"],
                    y=time_df["rolling_avg"],
                    mode="lines",
                    name="Moving Avg",
                    line=dict(color="#d4af37", width=3, shape="spline"),
                    showlegend=False
                ))
                
                fig_time.update_layout(title="Sentiment Over Time (Trend)")
                fig_time.add_hline(y=0.1, line_dash="dash", line_color="rgba(255,255,255,0.2)")
                fig_time.add_hline(y=-0.1, line_dash="dash", line_color="rgba(255,255,255,0.2)")
                style_chart(fig_time, height=400, x_title="Timestamp", y_title="Sentiment Score")
                st.plotly_chart(fig_time, use_container_width=True)

        show_cols = ["created_at", "source", "message", "polarity", "subjectivity", "sentiment"]
        st.dataframe(user_sent[show_cols].sort_values("created_at", ascending=False).head(25), width="stretch", height=360)

