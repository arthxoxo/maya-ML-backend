"""
Maya ML Dashboard — Feature Importance & User Behavior Visualization

Streamlit dashboard that visualizes:
  1. Global Feature Importance (SHAP + XGBoost)
  2. Per-User Feature Breakdown (SHAP waterfall)
  3. User Segmentation (K-Means clustering)
  4. Sentiment & Engagement Analysis
  5. Feature Correlation Heatmap
  6. GNN Insights (Graph Neural Network behavioral analysis)

Usage:
    source flink_venv/bin/activate
    streamlit run dashboard.py
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import xgboost as xgb
import shap
import json
from pathlib import Path
import warnings

warnings.filterwarnings("ignore")

# ── Page Config ──────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="Maya ML — Feature Intelligence",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ───────────────────────────────────────────────────────────────

st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

    .stApp {
        font-family: 'Inter', sans-serif;
    }

    .main-header {
        background: linear-gradient(135deg, #0f0c29, #302b63, #24243e);
        padding: 2rem 2.5rem;
        border-radius: 16px;
        margin-bottom: 2rem;
        color: white;
    }

    .main-header h1 {
        font-size: 2.2rem;
        font-weight: 700;
        margin: 0;
        background: linear-gradient(90deg, #00d2ff, #7b2ff7);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }

    .main-header p {
        color: #a0aec0;
        font-size: 1rem;
        margin-top: 0.5rem;
    }

    .metric-card {
        background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%);
        border: 1px solid rgba(123, 47, 247, 0.2);
        border-radius: 12px;
        padding: 1.2rem;
        text-align: center;
    }

    .metric-card h3 {
        font-size: 2rem;
        font-weight: 700;
        background: linear-gradient(90deg, #00d2ff, #7b2ff7);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin: 0;
    }

    .metric-card p {
        color: #a0aec0;
        font-size: 0.85rem;
        margin: 0.3rem 0 0;
    }

    .section-header {
        font-size: 1.4rem;
        font-weight: 600;
        color: #e2e8f0;
        margin: 2rem 0 1rem;
        padding-bottom: 0.5rem;
        border-bottom: 2px solid rgba(123, 47, 247, 0.3);
    }

    div[data-testid="stSidebar"] {
        background: linear-gradient(180deg, #0f0c29, #1a1a2e);
    }

    div[data-testid="stSidebar"] .stMarkdown {
        color: #cbd5e0;
    }

    .stSelectbox label, .stSlider label {
        color: #a0aec0 !important;
    }
</style>
""", unsafe_allow_html=True)


# ── Data Loading ─────────────────────────────────────────────────────────────

@st.cache_data
def load_data():
    df = pd.read_csv("/Users/arthxoxo/maya-ML-backend/user_feature_matrix.csv")
    return df


@st.cache_data
def get_feature_columns(df):
    """Get numeric feature columns (exclude identity cols)."""
    exclude = ["user_id", "user_name", "first_name", "last_name",
               "status", "type", "timezone"]
    return [c for c in df.columns if c not in exclude and df[c].dtype in [np.float64, np.int64, np.float32]]


@st.cache_resource
def compute_shap_values(df, feature_cols, target_col):
    """Train XGBoost and compute SHAP values."""
    X = df[feature_cols].fillna(0)
    y = df[target_col].fillna(0)

    model = xgb.XGBRegressor(
        n_estimators=100,
        max_depth=4,
        learning_rate=0.1,
        random_state=42,
        verbosity=0,
    )
    model.fit(X, y)

    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X)

    return model, explainer, shap_values, X


@st.cache_data
def compute_clusters(df, feature_cols, n_clusters=4):
    """K-Means clustering + PCA for visualization."""
    X = df[feature_cols].fillna(0)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    labels = kmeans.fit_predict(X_scaled)

    pca = PCA(n_components=2, random_state=42)
    coords = pca.fit_transform(X_scaled)

    return labels, coords, pca.explained_variance_ratio_


# ── Feature Groups ───────────────────────────────────────────────────────────

FEATURE_GROUPS = {
    "Message Behavior": [
        "total_messages_sent", "unique_sessions", "avg_message_length",
        "msg_length_std", "max_message_length", "question_ratio",
        "avg_conversation_depth", "tool_usage_rate", "total_input_tokens",
        "total_output_tokens", "total_cost_usd", "avg_response_time_sec",
        "median_response_time_sec",
    ],
    "Temporal Patterns": [
        "peak_usage_hour", "peak_usage_day", "morning_ratio",
        "afternoon_ratio", "evening_ratio", "night_ratio",
        "weekend_ratio", "active_days_count", "messages_per_active_day",
        "activity_std", "account_age_days", "longest_inactive_gap_days",
        "days_since_last_activity", "message_count_trend",
    ],
    "Sentiment": [
        "avg_sentiment", "sentiment_std", "min_sentiment", "max_sentiment",
        "negative_msg_ratio", "neutral_msg_ratio", "positive_msg_ratio",
        "avg_subjectivity", "sentiment_trend", "sentiment_volatility",
    ],
    "Session Engagement": [
        "total_sessions", "avg_session_duration_sec", "max_session_duration_sec",
        "total_session_duration_sec", "sessions_with_duration",
        "session_completion_rate", "avg_session_gap_hours",
        "min_session_gap_hours", "has_transcription_rate", "has_summary_rate",
        "sessions_per_week", "session_duration_trend",
    ],
    "Feedback": [
        "feedback_count", "feedback_avg_sentiment",
        "feedback_min_sentiment", "has_negative_feedback",
    ],
    "NLP Complexity": [
        "total_words", "unique_words", "vocabulary_richness",
        "avg_words_per_message", "emoji_usage_rate",
        "avg_sentence_length_chars", "short_msg_ratio", "long_msg_ratio",
    ],
    "Composite": ["engagement_score", "satisfaction_proxy"],
}

GROUP_COLORS = {
    "Message Behavior": "#00d2ff",
    "Temporal Patterns": "#7b2ff7",
    "Sentiment": "#ff6b6b",
    "Session Engagement": "#ffd93d",
    "Feedback": "#6bcb77",
    "NLP Complexity": "#4d96ff",
    "Composite": "#ff9a3c",
}


def get_feature_group(feature_name):
    for group, feats in FEATURE_GROUPS.items():
        if feature_name in feats:
            return group
    return "Other"


# ── Main App ─────────────────────────────────────────────────────────────────

def main():
    df = load_data()
    feature_cols = get_feature_columns(df)

    # ── Header ───────────────────────────────────────────────────────────
    st.markdown("""
    <div class="main-header">
        <h1>🧠 Maya ML — Feature Intelligence Dashboard</h1>
        <p>Per-user behavioral analysis • Sentiment trajectory • Feature importance • User segmentation</p>
    </div>
    """, unsafe_allow_html=True)

    # ── Sidebar ──────────────────────────────────────────────────────────
    with st.sidebar:
        st.markdown("### ⚙️ Configuration")

        target_col = st.selectbox(
            "Target for SHAP analysis",
            ["engagement_score", "satisfaction_proxy", "avg_sentiment"],
            index=0,
        )

        n_clusters = st.slider("Number of user clusters", 2, 8, 4)

        selected_user = st.selectbox(
            "Select user for deep-dive",
            df.sort_values("engagement_score", ascending=False)["user_name"].tolist(),
        )

        st.markdown("---")
        st.markdown(f"**Users:** {len(df)}")
        st.markdown(f"**Features:** {len(feature_cols)}")
        st.markdown(f"**Feature Groups:** {len(FEATURE_GROUPS)}")

    # ── KPI Cards ────────────────────────────────────────────────────────
    cols = st.columns(5)
    kpis = [
        (f"{len(df)}", "Total Users"),
        (f"{len(feature_cols)}", "Features Engineered"),
        (f"{df['total_messages_sent'].sum():,.0f}", "Total Messages"),
        (f"{df['avg_sentiment'].mean():.3f}", "Avg Sentiment"),
        (f"{df['engagement_score'].mean():.3f}", "Avg Engagement"),
    ]
    for col, (value, label) in zip(cols, kpis):
        col.markdown(f"""
        <div class="metric-card">
            <h3>{value}</h3>
            <p>{label}</p>
        </div>
        """, unsafe_allow_html=True)

    # ══════════════════════════════════════════════════════════════════════
    # TAB LAYOUT
    # ══════════════════════════════════════════════════════════════════════

    tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
        "📊 Feature Importance",
        "👤 Per-User Analysis",
        "🎯 User Segmentation",
        "💬 Sentiment Analysis",
        "🔥 Feature Correlation",
        "🧠 GNN Insights",
    ])

    # ══════════════════════════════════════════════════════════════════════
    # TAB 1: Global Feature Importance (SHAP)
    # ══════════════════════════════════════════════════════════════════════

    with tab1:
        st.markdown('<div class="section-header">Global Feature Importance (SHAP + XGBoost)</div>',
                    unsafe_allow_html=True)

        valid_features = [f for f in feature_cols if f != target_col]
        model, explainer, shap_values, X = compute_shap_values(df, valid_features, target_col)

        # Mean absolute SHAP values
        mean_shap = np.abs(shap_values).mean(axis=0)
        importance_df = pd.DataFrame({
            "feature": valid_features,
            "importance": mean_shap,
        }).sort_values("importance", ascending=True).tail(20)

        importance_df["group"] = importance_df["feature"].apply(get_feature_group)
        importance_df["color"] = importance_df["group"].map(GROUP_COLORS).fillna("#888")

        fig = px.bar(
            importance_df,
            x="importance",
            y="feature",
            color="group",
            color_discrete_map=GROUP_COLORS,
            orientation="h",
            title=f"Top 20 Features by SHAP Importance (Target: {target_col})",
            labels={"importance": "Mean |SHAP Value|", "feature": ""},
        )
        fig.update_layout(
            plot_bgcolor="rgba(0,0,0,0)",
            paper_bgcolor="rgba(0,0,0,0)",
            font=dict(color="#e2e8f0", family="Inter"),
            height=600,
            legend=dict(
                title="Feature Group",
                bgcolor="rgba(0,0,0,0.3)",
                font=dict(color="#cbd5e0"),
            ),
        )
        st.plotly_chart(fig, use_container_width=True)

        # Feature importance by group
        col1, col2 = st.columns(2)

        with col1:
            group_importance = importance_df.groupby("group")["importance"].sum().sort_values()
            fig2 = px.bar(
                x=group_importance.values,
                y=group_importance.index,
                orientation="h",
                title="Importance by Feature Group",
                color=group_importance.index,
                color_discrete_map=GROUP_COLORS,
            )
            fig2.update_layout(
                plot_bgcolor="rgba(0,0,0,0)",
                paper_bgcolor="rgba(0,0,0,0)",
                font=dict(color="#e2e8f0"),
                showlegend=False,
                height=400,
            )
            st.plotly_chart(fig2, use_container_width=True)

        with col2:
            # Feature group distribution pie chart
            all_importance = pd.DataFrame({
                "feature": valid_features,
                "importance": mean_shap,
            })
            all_importance["group"] = all_importance["feature"].apply(get_feature_group)
            group_totals = all_importance.groupby("group")["importance"].sum()

            fig3 = px.pie(
                values=group_totals.values,
                names=group_totals.index,
                title="Feature Group Contribution",
                color=group_totals.index,
                color_discrete_map=GROUP_COLORS,
                hole=0.4,
            )
            fig3.update_layout(
                plot_bgcolor="rgba(0,0,0,0)",
                paper_bgcolor="rgba(0,0,0,0)",
                font=dict(color="#e2e8f0"),
                height=400,
            )
            st.plotly_chart(fig3, use_container_width=True)

    # ══════════════════════════════════════════════════════════════════════
    # TAB 2: Per-User Analysis
    # ══════════════════════════════════════════════════════════════════════

    with tab2:
        st.markdown(f'<div class="section-header">User Deep-Dive: {selected_user}</div>',
                    unsafe_allow_html=True)

        user_row = df[df["user_name"] == selected_user].iloc[0]
        user_idx = df[df["user_name"] == selected_user].index[0]

        # User stats cards
        ucols = st.columns(4)
        user_kpis = [
            (f"{user_row.get('total_messages_sent', 0):.0f}", "Messages Sent"),
            (f"{user_row.get('total_sessions', 0):.0f}", "Sessions"),
            (f"{user_row.get('avg_sentiment', 0):.3f}", "Avg Sentiment"),
            (f"{user_row.get('engagement_score', 0):.3f}", "Engagement Score"),
        ]
        for col, (v, l) in zip(ucols, user_kpis):
            col.markdown(f'<div class="metric-card"><h3>{v}</h3><p>{l}</p></div>',
                         unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)

        # Per-user SHAP values
        user_shap = shap_values[user_idx]
        user_shap_df = pd.DataFrame({
            "feature": valid_features,
            "shap_value": user_shap,
            "abs_shap": np.abs(user_shap),
            "feature_value": X.iloc[user_idx].values,
        }).sort_values("abs_shap", ascending=True).tail(15)

        user_shap_df["direction"] = user_shap_df["shap_value"].apply(
            lambda x: "Increases" if x > 0 else "Decreases"
        )
        user_shap_df["group"] = user_shap_df["feature"].apply(get_feature_group)

        fig4 = px.bar(
            user_shap_df,
            x="shap_value",
            y="feature",
            color="direction",
            color_discrete_map={"Increases": "#6bcb77", "Decreases": "#ff6b6b"},
            orientation="h",
            title=f"What Drives {selected_user}'s {target_col}?",
            labels={"shap_value": f"Impact on {target_col}", "feature": ""},
        )
        fig4.update_layout(
            plot_bgcolor="rgba(0,0,0,0)",
            paper_bgcolor="rgba(0,0,0,0)",
            font=dict(color="#e2e8f0"),
            height=500,
        )
        st.plotly_chart(fig4, use_container_width=True)

        # User feature radar
        col1, col2 = st.columns(2)

        with col1:
            radar_features = ["engagement_score", "avg_sentiment", "question_ratio",
                              "tool_usage_rate", "vocabulary_richness", "weekend_ratio",
                              "session_completion_rate"]
            radar_features = [f for f in radar_features if f in df.columns]

            user_vals = []
            for f in radar_features:
                col_max = df[f].max()
                user_vals.append(user_row[f] / col_max if col_max > 0 else 0)

            fig5 = go.Figure()
            fig5.add_trace(go.Scatterpolar(
                r=user_vals + [user_vals[0]],
                theta=radar_features + [radar_features[0]],
                fill='toself',
                fillcolor='rgba(123, 47, 247, 0.2)',
                line=dict(color='#7b2ff7', width=2),
                name=selected_user,
            ))
            fig5.update_layout(
                polar=dict(
                    bgcolor="rgba(0,0,0,0)",
                    radialaxis=dict(visible=True, range=[0, 1], gridcolor="rgba(255,255,255,0.1)"),
                    angularaxis=dict(gridcolor="rgba(255,255,255,0.1)"),
                ),
                plot_bgcolor="rgba(0,0,0,0)",
                paper_bgcolor="rgba(0,0,0,0)",
                font=dict(color="#e2e8f0"),
                title="User Profile Radar (normalized)",
                height=400,
                showlegend=False,
            )
            st.plotly_chart(fig5, use_container_width=True)

        with col2:
            # User's temporal pattern
            time_features = ["morning_ratio", "afternoon_ratio", "evening_ratio", "night_ratio"]
            time_features = [f for f in time_features if f in df.columns]
            time_vals = [user_row[f] for f in time_features]
            time_labels = ["Morning\n6AM-12PM", "Afternoon\n12PM-6PM", "Evening\n6PM-10PM", "Night\n10PM-6AM"]

            fig6 = go.Figure()
            fig6.add_trace(go.Bar(
                x=time_labels,
                y=time_vals,
                marker=dict(
                    color=time_vals,
                    colorscale=[[0, '#302b63'], [1, '#00d2ff']],
                ),
            ))
            fig6.update_layout(
                title=f"{selected_user}'s Activity by Time of Day",
                plot_bgcolor="rgba(0,0,0,0)",
                paper_bgcolor="rgba(0,0,0,0)",
                font=dict(color="#e2e8f0"),
                yaxis_title="Message Ratio",
                height=400,
            )
            st.plotly_chart(fig6, use_container_width=True)

    # ══════════════════════════════════════════════════════════════════════
    # TAB 3: User Segmentation
    # ══════════════════════════════════════════════════════════════════════

    with tab3:
        st.markdown('<div class="section-header">User Segmentation (K-Means + PCA)</div>',
                    unsafe_allow_html=True)

        labels, coords, var_ratio = compute_clusters(df, feature_cols, n_clusters)
        df["cluster"] = labels

        cluster_names = {
            0: "Power Users",
            1: "Casual Users",
            2: "New/Exploring",
            3: "Moderate Users",
            4: "Inactive",
            5: "Heavy Session",
            6: "Feedback Givers",
            7: "Night Owls",
        }

        scatter_df = pd.DataFrame({
            "PC1": coords[:, 0],
            "PC2": coords[:, 1],
            "user_name": df["user_name"],
            "cluster": [cluster_names.get(l, f"Cluster {l}") for l in labels],
            "engagement": df["engagement_score"],
            "messages": df["total_messages_sent"],
            "sentiment": df["avg_sentiment"],
        })

        cluster_colors = ["#00d2ff", "#7b2ff7", "#ff6b6b", "#ffd93d",
                          "#6bcb77", "#4d96ff", "#ff9a3c", "#e056fd"]

        fig7 = px.scatter(
            scatter_df,
            x="PC1", y="PC2",
            color="cluster",
            size="engagement",
            hover_data=["user_name", "messages", "sentiment"],
            title=f"User Clusters (PCA explains {sum(var_ratio)*100:.1f}% variance)",
            color_discrete_sequence=cluster_colors[:n_clusters],
        )
        fig7.update_layout(
            plot_bgcolor="rgba(0,0,0,0)",
            paper_bgcolor="rgba(0,0,0,0)",
            font=dict(color="#e2e8f0"),
            height=550,
        )
        fig7.update_traces(marker=dict(line=dict(width=1, color='rgba(255,255,255,0.3)')))
        st.plotly_chart(fig7, use_container_width=True)

        # Cluster stats
        st.markdown("#### Cluster Profiles")
        cluster_stats_cols = ["total_messages_sent", "total_sessions", "avg_sentiment",
                              "engagement_score", "active_days_count", "question_ratio"]
        cluster_stats_cols = [c for c in cluster_stats_cols if c in df.columns]

        cluster_summary = df.groupby("cluster")[cluster_stats_cols].mean().round(3)
        cluster_summary.index = [cluster_names.get(i, f"Cluster {i}") for i in cluster_summary.index]
        st.dataframe(cluster_summary, use_container_width=True)

    # ══════════════════════════════════════════════════════════════════════
    # TAB 4: Sentiment Analysis
    # ══════════════════════════════════════════════════════════════════════

    with tab4:
        st.markdown('<div class="section-header">Sentiment Analysis Across Users</div>',
                    unsafe_allow_html=True)

        col1, col2 = st.columns(2)

        with col1:
            # Sentiment distribution
            fig8 = px.histogram(
                df, x="avg_sentiment",
                nbins=20,
                title="Sentiment Distribution Across Users",
                color_discrete_sequence=["#7b2ff7"],
                labels={"avg_sentiment": "Average Sentiment Score"},
            )
            fig8.update_layout(
                plot_bgcolor="rgba(0,0,0,0)",
                paper_bgcolor="rgba(0,0,0,0)",
                font=dict(color="#e2e8f0"),
                height=400,
            )
            st.plotly_chart(fig8, use_container_width=True)

        with col2:
            # Sentiment vs Engagement scatter
            fig9 = px.scatter(
                df,
                x="avg_sentiment",
                y="engagement_score",
                size="total_messages_sent",
                hover_data=["user_name"],
                title="Sentiment vs Engagement",
                color="satisfaction_proxy",
                color_continuous_scale="Viridis",
            )
            fig9.update_layout(
                plot_bgcolor="rgba(0,0,0,0)",
                paper_bgcolor="rgba(0,0,0,0)",
                font=dict(color="#e2e8f0"),
                height=400,
            )
            st.plotly_chart(fig9, use_container_width=True)

        # Sentiment polarity breakdown per user (top 15)
        top_users = df.nlargest(15, "total_messages_sent")
        polarity_df = top_users[["user_name", "positive_msg_ratio", "neutral_msg_ratio", "negative_msg_ratio"]]
        polarity_melted = polarity_df.melt(
            id_vars="user_name",
            var_name="polarity",
            value_name="ratio",
        )
        polarity_melted["polarity"] = polarity_melted["polarity"].map({
            "positive_msg_ratio": "Positive",
            "neutral_msg_ratio": "Neutral",
            "negative_msg_ratio": "Negative",
        })

        fig10 = px.bar(
            polarity_melted,
            x="user_name",
            y="ratio",
            color="polarity",
            color_discrete_map={
                "Positive": "#6bcb77",
                "Neutral": "#ffd93d",
                "Negative": "#ff6b6b",
            },
            title="Sentiment Breakdown — Top 15 Active Users",
            barmode="stack",
        )
        fig10.update_layout(
            plot_bgcolor="rgba(0,0,0,0)",
            paper_bgcolor="rgba(0,0,0,0)",
            font=dict(color="#e2e8f0"),
            height=450,
            xaxis_tickangle=-45,
        )
        st.plotly_chart(fig10, use_container_width=True)

    # ══════════════════════════════════════════════════════════════════════
    # TAB 5: Feature Correlation
    # ══════════════════════════════════════════════════════════════════════

    with tab5:
        st.markdown('<div class="section-header">Feature Correlation Heatmap</div>',
                    unsafe_allow_html=True)

        # Select key features for readable heatmap
        key_features = [
            "total_messages_sent", "total_sessions", "avg_message_length",
            "question_ratio", "tool_usage_rate", "avg_sentiment",
            "sentiment_trend", "engagement_score", "active_days_count",
            "sessions_per_week", "vocabulary_richness", "satisfaction_proxy",
            "avg_response_time_sec", "session_completion_rate",
            "morning_ratio", "weekend_ratio",
        ]
        key_features = [f for f in key_features if f in df.columns]

        corr = df[key_features].corr()

        fig11 = px.imshow(
            corr,
            x=key_features,
            y=key_features,
            color_continuous_scale="RdBu_r",
            zmin=-1, zmax=1,
            title="Feature Correlation Matrix (Key Features)",
        )
        fig11.update_layout(
            plot_bgcolor="rgba(0,0,0,0)",
            paper_bgcolor="rgba(0,0,0,0)",
            font=dict(color="#e2e8f0", size=10),
            height=650,
        )
        st.plotly_chart(fig11, use_container_width=True)

        # Top correlated pairs
        st.markdown("#### Top Correlated Feature Pairs")
        corr_pairs = []
        for i in range(len(key_features)):
            for j in range(i + 1, len(key_features)):
                corr_pairs.append({
                    "Feature 1": key_features[i],
                    "Feature 2": key_features[j],
                    "Correlation": round(corr.iloc[i, j], 3),
                })
        corr_pairs_df = pd.DataFrame(corr_pairs)
        corr_pairs_df["abs_corr"] = corr_pairs_df["Correlation"].abs()
        top_corr = corr_pairs_df.nlargest(10, "abs_corr").drop(columns="abs_corr")
        st.dataframe(top_corr, use_container_width=True, hide_index=True)

    # ══════════════════════════════════════════════════════════════════════
    # TAB 6: GNN Insights
    # ══════════════════════════════════════════════════════════════════════

    with tab6:
        st.markdown('<div class="section-header">🧠 GNN Behavioral Intelligence</div>',
                    unsafe_allow_html=True)

        # ---- Load GNN artifacts ----
        gnn_dir = Path("/Users/arthxoxo/maya-ML-backend")
        metrics_path = gnn_dir / "gnn_training_metrics.json"
        importance_path = gnn_dir / "gnn_feature_importance.csv"
        embeddings_path = gnn_dir / "gnn_user_embeddings.csv"
        predictions_dir = gnn_dir / "gnn_predictions"

        has_gnn = metrics_path.exists() and importance_path.exists()

        if not has_gnn:
            st.warning("⚠️ GNN model not yet trained. Run `python gnn_train.py` first.")
        else:
            # Load GNN data
            with open(metrics_path) as f:
                gnn_metrics = json.load(f)
            gnn_importance = pd.read_csv(importance_path)

            gnn_embeddings = None
            if embeddings_path.exists():
                gnn_embeddings = pd.read_csv(embeddings_path)

            # ---- GNN KPI Cards ----
            gcols = st.columns(5)
            gnn_kpis = [
                (f"{gnn_metrics.get('all_r2', 0):.4f}", "GNN R² Score"),
                (f"{gnn_metrics.get('all_mse', 0):.4f}", "Test MSE"),
                (f"{gnn_metrics.get('all_mae', 0):.4f}", "Test MAE"),
                (f"{gnn_metrics.get('best_epoch', 0)}", "Best Epoch"),
                (f"{gnn_metrics.get('training_time_sec', 0):.1f}s", "Train Time"),
            ]
            for col, (v, l) in zip(gcols, gnn_kpis):
                col.markdown(f'<div class="metric-card"><h3>{v}</h3><p>{l}</p></div>',
                             unsafe_allow_html=True)

            st.markdown("<br>", unsafe_allow_html=True)

            # ---- 6A: GNN vs XGBoost Feature Importance Comparison ----
            st.markdown("#### 🔀 GNN vs XGBoost Feature Importance")

            # GNN global importance
            gnn_global = gnn_importance.groupby("feature")["importance_raw"].mean()
            gnn_global = gnn_global.sort_values(ascending=False).head(15)

            # XGBoost SHAP importance (from tab1 computation)
            valid_features_for_gnn = [f for f in feature_cols if f != target_col]
            _, _, shap_vals_gnn, _ = compute_shap_values(df, valid_features_for_gnn, target_col)
            xgb_global = pd.Series(
                np.abs(shap_vals_gnn).mean(axis=0),
                index=valid_features_for_gnn
            ).sort_values(ascending=False).head(15)

            # Merge for comparison
            comp_features = list(set(gnn_global.index) | set(xgb_global.index))
            comp_df = pd.DataFrame({
                "feature": comp_features,
                "GNN Importance": [gnn_global.get(f, 0) for f in comp_features],
                "XGBoost SHAP": [xgb_global.get(f, 0) for f in comp_features],
            })

            # Normalize to same scale for comparison
            for col_name in ["GNN Importance", "XGBoost SHAP"]:
                max_val = comp_df[col_name].max()
                if max_val > 0:
                    comp_df[col_name] = comp_df[col_name] / max_val

            comp_df = comp_df.sort_values("GNN Importance", ascending=True).tail(15)

            fig_comp = go.Figure()
            fig_comp.add_trace(go.Bar(
                y=comp_df["feature"],
                x=comp_df["GNN Importance"],
                name="GNN (Graph)",
                orientation="h",
                marker_color="#7b2ff7",
            ))
            fig_comp.add_trace(go.Bar(
                y=comp_df["feature"],
                x=comp_df["XGBoost SHAP"],
                name="XGBoost (Tabular)",
                orientation="h",
                marker_color="#00d2ff",
            ))
            fig_comp.update_layout(
                barmode="group",
                title="GNN vs XGBoost: Normalized Feature Importance (Top 15)",
                xaxis_title="Normalized Importance",
                plot_bgcolor="rgba(0,0,0,0)",
                paper_bgcolor="rgba(0,0,0,0)",
                font=dict(color="#e2e8f0", family="Inter"),
                height=550,
                legend=dict(bgcolor="rgba(0,0,0,0.3)", font=dict(color="#cbd5e0")),
            )
            st.plotly_chart(fig_comp, use_container_width=True)

            # ---- 6B: Per-User GNN Explanation ----
            st.markdown(f"#### 🔍 GNN Explanation for {selected_user}")

            user_gnn_imp = gnn_importance[gnn_importance["user_name"] == selected_user]
            if len(user_gnn_imp) > 0:
                user_gnn_top = user_gnn_imp.nlargest(10, "importance_normalized")

                predicted_eng = user_gnn_top.iloc[0]["predicted_engagement"]
                actual_eng = df.loc[df["user_name"] == selected_user, "engagement_score"].values
                actual_eng = actual_eng[0] if len(actual_eng) > 0 else 0

                ucols2 = st.columns(3)
                ucols2[0].markdown(
                    f'<div class="metric-card"><h3>{predicted_eng:.4f}</h3>'
                    f'<p>GNN Predicted Engagement</p></div>',
                    unsafe_allow_html=True
                )
                ucols2[1].markdown(
                    f'<div class="metric-card"><h3>{actual_eng:.4f}</h3>'
                    f'<p>Actual Engagement</p></div>',
                    unsafe_allow_html=True
                )
                error = abs(predicted_eng - actual_eng)
                ucols2[2].markdown(
                    f'<div class="metric-card"><h3>{error:.4f}</h3>'
                    f'<p>Prediction Error</p></div>',
                    unsafe_allow_html=True
                )

                st.markdown("<br>", unsafe_allow_html=True)

                fig_user_gnn = px.bar(
                    user_gnn_top,
                    x="importance_normalized",
                    y="feature",
                    orientation="h",
                    title=f"Top Features Driving {selected_user}'s GNN Prediction",
                    labels={"importance_normalized": "Normalized Importance", "feature": ""},
                    color="importance_normalized",
                    color_continuous_scale=[[0, "#302b63"], [1, "#7b2ff7"]],
                )
                fig_user_gnn.update_layout(
                    plot_bgcolor="rgba(0,0,0,0)",
                    paper_bgcolor="rgba(0,0,0,0)",
                    font=dict(color="#e2e8f0"),
                    height=450,
                    showlegend=False,
                )
                st.plotly_chart(fig_user_gnn, use_container_width=True)
            else:
                st.info(f"No GNN feature importance data found for {selected_user}")

            # ---- 6C: Training Loss Curve ----
            col1, col2 = st.columns(2)

            with col1:
                st.markdown("#### 📉 Training History")
                history = gnn_metrics.get("history", {})
                if history:
                    epochs = list(range(1, len(history.get("train_loss", [])) + 1))
                    fig_loss = go.Figure()
                    fig_loss.add_trace(go.Scatter(
                        x=epochs,
                        y=history.get("train_loss", []),
                        name="Train Loss",
                        line=dict(color="#7b2ff7", width=2),
                    ))
                    fig_loss.add_trace(go.Scatter(
                        x=epochs,
                        y=history.get("val_mse", []),
                        name="Val MSE",
                        line=dict(color="#00d2ff", width=2),
                    ))
                    # Mark best epoch
                    best_ep = gnn_metrics.get("best_epoch", 0)
                    if best_ep > 0 and best_ep <= len(history.get("val_mse", [])):
                        fig_loss.add_vline(
                            x=best_ep, line_dash="dash",
                            line_color="#ffd93d",
                            annotation_text=f"Best: Epoch {best_ep}",
                            annotation_font_color="#ffd93d",
                        )
                    fig_loss.update_layout(
                        title="Loss & Validation MSE",
                        xaxis_title="Epoch",
                        yaxis_title="Loss / MSE",
                        plot_bgcolor="rgba(0,0,0,0)",
                        paper_bgcolor="rgba(0,0,0,0)",
                        font=dict(color="#e2e8f0"),
                        height=400,
                        legend=dict(bgcolor="rgba(0,0,0,0.3)"),
                    )
                    st.plotly_chart(fig_loss, use_container_width=True)

            with col2:
                st.markdown("#### 🌐 GNN User Embeddings (PCA)")
                if gnn_embeddings is not None:
                    embed_cols = [c for c in gnn_embeddings.columns if c.startswith("gnn_embed_")]
                    if len(embed_cols) >= 2:
                        from sklearn.decomposition import PCA as PCA2
                        pca_gnn = PCA2(n_components=2, random_state=42)
                        coords_gnn = pca_gnn.fit_transform(gnn_embeddings[embed_cols].fillna(0))

                        embed_vis = pd.DataFrame({
                            "PC1": coords_gnn[:, 0],
                            "PC2": coords_gnn[:, 1],
                            "user_name": gnn_embeddings["user_name"],
                        })

                        # Merge engagement for coloring
                        embed_vis = embed_vis.merge(
                            df[["user_name", "engagement_score", "total_messages_sent"]],
                            on="user_name", how="left"
                        )

                        fig_embed = px.scatter(
                            embed_vis,
                            x="PC1", y="PC2",
                            color="engagement_score",
                            size="total_messages_sent",
                            hover_data=["user_name"],
                            title=f"User Embeddings (PCA: {pca_gnn.explained_variance_ratio_.sum()*100:.1f}% var)",
                            color_continuous_scale="Viridis",
                        )
                        fig_embed.update_layout(
                            plot_bgcolor="rgba(0,0,0,0)",
                            paper_bgcolor="rgba(0,0,0,0)",
                            font=dict(color="#e2e8f0"),
                            height=400,
                        )
                        fig_embed.update_traces(
                            marker=dict(line=dict(width=1, color="rgba(255,255,255,0.3)"))
                        )
                        st.plotly_chart(fig_embed, use_container_width=True)

            # ---- 6D: GNN Predictions Table ----
            st.markdown("#### 📋 All User Predictions")

            # Build predictions table from importance data
            if len(gnn_importance) > 0:
                pred_table = gnn_importance.groupby(["user_id", "user_name"]).agg(
                    predicted_engagement=("predicted_engagement", "first"),
                    top_feature=("importance_normalized", lambda x: gnn_importance.loc[
                        x.idxmax(), "feature"
                    ] if len(x) > 0 else "N/A"),
                ).reset_index()

                # Merge actual engagement
                pred_table = pred_table.merge(
                    df[["user_name", "engagement_score"]],
                    on="user_name", how="left"
                )
                pred_table["error"] = abs(
                    pred_table["predicted_engagement"] - pred_table["engagement_score"]
                )
                pred_table = pred_table.sort_values("predicted_engagement", ascending=False)

                st.dataframe(
                    pred_table[[
                        "user_name", "predicted_engagement",
                        "engagement_score", "error", "top_feature"
                    ]].rename(columns={
                        "user_name": "User",
                        "predicted_engagement": "GNN Predicted",
                        "engagement_score": "Actual",
                        "error": "Abs Error",
                        "top_feature": "Top Feature",
                    }),
                    use_container_width=True,
                    hide_index=True,
                )

            # ---- 6E: Streaming Predictions Feed ----
            if predictions_dir.exists():
                csv_files = list(predictions_dir.glob("*.csv"))
                if csv_files:
                    st.markdown("#### ⚡ Live Streaming Predictions")
                    all_preds = []
                    for csv_file in csv_files[-5:]:  # Last 5 files
                        try:
                            pred_df = pd.read_csv(csv_file, header=None,
                                                  names=["user_name", "event_type", "gnn_prediction"])
                            all_preds.append(pred_df)
                        except Exception:
                            pass
                    if all_preds:
                        stream_df = pd.concat(all_preds, ignore_index=True).tail(20)
                        st.dataframe(stream_df, use_container_width=True, hide_index=True)
                    else:
                        st.info("📡 No streaming predictions yet. "
                                "Run `python flink_gnn_inference_job.py` to start.")
                else:
                    st.info("📡 No streaming predictions yet. "
                            "Run `python flink_gnn_inference_job.py` to start.")


if __name__ == "__main__":
    main()
