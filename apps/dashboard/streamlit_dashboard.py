"""
Main entry point for Maya Behavioral Intelligence Dashboard.
Routes to specific page modules.
"""

from __future__ import annotations

import streamlit as st
import pandas as pd
import numpy as np
from pathlib import Path
import sys

# Ensure project-root imports resolve
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from apps.dashboard.shared import style_app, get_dashboard_last_updated_label, get_data_refresh_nonce
from apps.dashboard.data_loaders import (
    load_whatsapp_sentiment_messages,
    load_user_directory,
    load_sentiment_table,
    load_outputs,
)

# Import Pages
from apps.dashboard import page_global_insights
from apps.dashboard import page_per_user_analysis
from apps.dashboard import page_rag_roadmap
from apps.dashboard import page_global_sentiment
from apps.dashboard import page_per_user_sentiment


def main() -> None:
    style_app()
    refresh_nonce = get_data_refresh_nonce()

    st.markdown("""
        <div style="margin-bottom: 0.5rem;">
            <div style="display: flex; align-items: center; gap: 14px; margin-bottom: 6px;">
                <span style="font-size: 2rem;">🔱</span>
                <h1 style="margin: 0; padding: 0; font-family: 'Outfit', sans-serif; font-size: 2.2rem; font-weight: 700;
                    background: linear-gradient(135deg, #f1f5f9 0%, #d4af37 100%);
                    -webkit-background-clip: text; -webkit-text-fill-color: transparent; background-clip: text;">
                    Maya Behavioral Intelligence
                </h1>
            </div>
            <div style="display: flex; align-items: center; gap: 16px; margin-left: 3.2rem;">
                <span style="color: #64748b; font-size: 0.82rem; font-family: 'Inter', sans-serif;">
                    User-level feature importance and sentiment insights from your trained GNN outputs
                </span>
                <span style="color: rgba(212,175,55,0.4);">•</span>
                <span style="color: #4a5568; font-size: 0.78rem; font-family: 'Inter', sans-serif;">
                    """ + f"Last Updated: {get_dashboard_last_updated_label()}" + """
                </span>
            </div>
        </div>
    """, unsafe_allow_html=True)

    page = st.sidebar.radio(
        "Page",
        [
            "Global Insights",
            "Per-User Analysis",
            "RAG Roadmap Signals",
            "Global Sentiment Analysis",
            "Per-User Sentiment Analysis",
        ],
    )


    # Route to appropriate page module
    if page == "Global Insights":
        scores, _, _ = load_outputs(refresh_nonce)
        sentiment_df = load_sentiment_table(refresh_nonce)
        page_global_insights.render(scores, sentiment_df)
        
    elif page == "Per-User Analysis":
        scores, _, per_user_imp = load_outputs(refresh_nonce)
        sentiment_df = load_sentiment_table(refresh_nonce)
        user_directory = load_user_directory()
        
        display_map = {}
        if not user_directory.empty:
            for _, r in user_directory.iterrows():
                display_map[int(r["user_id"])] = str(r["display_name"])
                
        score_users = sorted(scores["user_id"].unique().tolist()) if not scores.empty else []
        page_per_user_analysis.render(refresh_nonce, display_map, score_users, scores, per_user_imp, sentiment_df)
        
    elif page == "RAG Roadmap Signals":
        page_rag_roadmap.render(refresh_nonce)
        
    elif page == "Global Sentiment Analysis":
        wa = load_whatsapp_sentiment_messages(refresh_nonce)
        user_directory = load_user_directory()
        name_map = {}
        if not user_directory.empty:
            for _, row in user_directory.iterrows():
                name_map[int(row["user_id"])] = str(row["display_name"])
        page_global_sentiment.render(wa, user_directory, name_map, refresh_nonce)
        
    elif page == "Per-User Sentiment Analysis":
        wa = load_whatsapp_sentiment_messages(refresh_nonce)
        user_directory = load_user_directory()
        name_map = {}
        if not user_directory.empty:
            for _, row in user_directory.iterrows():
                name_map[int(row["user_id"])] = str(row["display_name"])
        page_per_user_sentiment.render(wa, user_directory, name_map, refresh_nonce)

if __name__ == "__main__":
    main()
