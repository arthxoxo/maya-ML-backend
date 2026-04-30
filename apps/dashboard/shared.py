"""
Shared constants, path definitions, color palettes, styling, and reusable UI components
for the Maya Behavioral Intelligence dashboard.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Tuple
import re
from collections import Counter
import math
import os
import json
import subprocess
import sys
from datetime import datetime

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import plotly.io as pio
import streamlit as st

# Ensure project-root imports (e.g., app_config) resolve even when launching from nested dirs.
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import app_config as cfg

# ═══════════════════════════════════════════════
#  PATH CONFIGURATION
# ═══════════════════════════════════════════════
BASE_DIR = cfg.BASE_DIR
GNN_OUTPUT_DIR = cfg.GNN_OUTPUT_DIR
GNN_PREPROCESSED_DIR = cfg.GNN_PREPROCESSED_DIR
SECRET_DATA_DIR = cfg.SECRET_DATA_DIR
FLINK_ENGINEERED_DIR = getattr(cfg, "FLINK_ENGINEERED_DIR", BASE_DIR / "flink_engineered")

EMBEDDINGS_ARTIFACT_DIR = getattr(
    cfg,
    "EMBEDDINGS_ARTIFACT_DIR",
    getattr(cfg, "EMBEDDING_ARTIFACT_DIR", BASE_DIR / "artifacts" / "embeddings"),
)
XGB_ARTIFACT_DIR = getattr(cfg, "XGB_ARTIFACT_DIR", BASE_DIR / "artifacts" / "xgb")
PERSONA_ARTIFACT_DIR = getattr(cfg, "PERSONA_ARTIFACT_DIR", BASE_DIR / "artifacts" / "persona")
SENTIMENT_ARTIFACT_DIR = getattr(cfg, "SENTIMENT_ARTIFACT_DIR", BASE_DIR / "artifacts" / "sentiment")


def _first_existing_path(*candidates: Path) -> Path:
    for p in candidates:
        if p.exists():
            return p
    return candidates[0]


OUTPUT_DIR = GNN_OUTPUT_DIR
PREPROCESSED_DIR = GNN_PREPROCESSED_DIR
XGB_SHAP_PLOT_PATH = XGB_ARTIFACT_DIR / "shap_summary.png"
XGB_SHAP_IMPORTANCE_PATH = _first_existing_path(
    XGB_ARTIFACT_DIR / "xgb_embedding_feature_importance.csv",
    XGB_ARTIFACT_DIR / "xgb_embedding_feature_importance_merged.csv",
)
XGB_PREDICTIONS_PATH = XGB_ARTIFACT_DIR / "xgb_user_predictions.csv"
XGB_MODEL_PATH_CANDIDATES = [
    XGB_ARTIFACT_DIR / "xgb_model.json",
    XGB_ARTIFACT_DIR / "xgb_model.ubj",
    XGB_ARTIFACT_DIR / "xgb_model.pkl",
]
EMBEDDING_LABELS_PATH = EMBEDDINGS_ARTIFACT_DIR / "embedding_dimension_labels.csv"
USER_EMBEDDINGS_PATH = _first_existing_path(
    EMBEDDINGS_ARTIFACT_DIR / "user_embeddings.csv",
    OUTPUT_DIR / "user_embeddings.csv",
)
PERSONA_TABLE_PATH = PERSONA_ARTIFACT_DIR / "user_persona_table.csv"
PERSONA_PROFILE_PATH = PERSONA_ARTIFACT_DIR / "persona_profiles.csv"
PERSONA_IMPORTANCE_PATH = PERSONA_ARTIFACT_DIR / "persona_feature_importance.csv"
PERSONA_SHAP_PLOT_PATH = PERSONA_ARTIFACT_DIR / "persona_shap_summary.png"
PERSONA_USER_SHAP_PATH = PERSONA_ARTIFACT_DIR / "persona_user_feature_contributions.csv"
SENTIMENT_SCORES_PATH = _first_existing_path(
    SENTIMENT_ARTIFACT_DIR / "sentiment_scores.csv",
    BASE_DIR / "sentiment_scores.csv",
)
GRU_MOOD_SWING_SUMMARY_PATH = SENTIMENT_ARTIFACT_DIR / "gru_mood_swing_summary.csv"
GRU_MOOD_TRAINING_REPORT_PATH = SENTIMENT_ARTIFACT_DIR / "gru_mood_training_report.csv"
SESSIONS_SOURCE_PATH = _first_existing_path(
    SECRET_DATA_DIR / "sessions.csv",
    BASE_DIR / "sessions.csv",
)
RAW_USERS_PATH = _first_existing_path(SECRET_DATA_DIR / "users.csv", BASE_DIR / "users.csv")
RAW_SESSIONS_PATH = _first_existing_path(SECRET_DATA_DIR / "sessions.csv", BASE_DIR / "sessions.csv")
RAW_MESSAGES_PATH = _first_existing_path(SECRET_DATA_DIR / "whatsapp_messages.csv", BASE_DIR / "whatsapp_messages.csv")
HF_SENTIMENT_MODEL = "cardiffnlp/twitter-roberta-base-sentiment-latest"
HF_IRONY_MODEL = "cardiffnlp/twitter-roberta-base-irony"
REDIS_KEY_PREFIX = os.getenv("MAYA_REDIS_PREFIX", "maya:dashboard")

# ═══════════════════════════════════════════════
#  COLOR PALETTES & DESIGN TOKENS
# ═══════════════════════════════════════════════
SENTIMENT_COLORS = {
    "positive": "#2ECC71",  # Emerald
    "neutral": "#94A3B8",   # Slate
    "negative": "#E74C3C",  # Ruby
}

SENTIMENT_DIVERGING_SCALE = [
    [0.00, "#E74C3C"],
    [0.35, "#F39C12"],
    [0.50, "#1E293B"],
    [0.65, "#3498DB"],
    [1.00, "#2ECC71"],
]

ACCENT_PRIMARY = "#D4AF37"  # Metallic Gold
ACCENT_SECONDARY = "#BD9354" # Champagne
CHART_PAPER_BG = "rgba(0,0,0,0)"
CHART_PLOT_BG = "rgba(0,0,0,0)"
GRID_COLOR = "rgba(255,255,255,0.05)"
PERSONA_COLORS = ["#D4AF37", "#2C3E50", "#7F8C8D", "#16A085", "#2980B9", "#8E44AD", "#F39C12"]
RISK_COLORS = {"High": "#E74C3C", "Medium": "#F39C12", "Low": "#2ECC71"}

# Global Plotly Luxury Dark Theme
pio.templates["maya_luxury"] = go.layout.Template(
    layout=go.Layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#F8FAFC", family="Inter, sans-serif"),
        xaxis=dict(
            gridcolor="rgba(255,255,255,0.05)", 
            zerolinecolor="rgba(255,255,255,0.1)",
            title_font=dict(color="#F1F5F9", size=14),
            tickfont=dict(color="#CBD5E1", size=12)
        ),
        yaxis=dict(
            gridcolor="rgba(255,255,255,0.05)", 
            zerolinecolor="rgba(255,255,255,0.1)",
            title_font=dict(color="#F1F5F9", size=14),
            tickfont=dict(color="#CBD5E1", size=12)
        ),
        hoverlabel=dict(bgcolor="#1C1F26", font_size=13, font_family="Inter", font_color="#F8FAFC"),
        colorway=PERSONA_COLORS,
    )
)
pio.templates.default = "plotly_dark+maya_luxury"

# ═══════════════════════════════════════════════
#  REGEX PATTERNS & WORD SETS
# ═══════════════════════════════════════════════
GEO_NOISE_PATTERN = r"latitude|longitude|timezone|country|city|state|zip|postal|region|geo|location"

STOPWORDS = {
    "a", "an", "and", "are", "as", "at", "be", "by", "for", "from", "has", "he", "i", "in", "is", "it",
    "its", "me", "my", "of", "on", "or", "our", "she", "that", "the", "their", "them", "they", "this", "to",
    "us", "we", "were", "will", "with", "you", "your", "yours", "im", "ive", "dont", "cant", "did", "do", "does",
    "was", "what", "when", "where", "which", "who", "why", "how", "am", "not", "but", "if", "then", "than",
}

FILLER_WORDS = {
    "can", "could", "would", "should", "have", "had", "having", "please", "kindly", "just", "like", "really",
    "maybe", "also", "there", "here", "bot", "assistant", "chatbot", "hey", "hello", "hi", "thanks", "thank",
}

ACTION_VERBS = {
    "analyze", "book", "build", "calculate", "check", "compare", "create", "debug", "draft", "explain", "find",
    "fix", "generate", "help", "list", "optimize", "plan", "prepare", "recommend", "remind", "schedule", "search",
    "send", "show", "solve", "summarize", "track", "translate", "update", "write",
}

TASK_PATTERNS = [
    re.compile(r"\b(?:can you|could you|would you|please|kindly)\s+([a-z][a-z'\s]{2,80})"),
    re.compile(r"\b(?:i need to|i want to|help me to|help me|let me|need to|want to)\s+([a-z][a-z'\s]{2,80})"),
    re.compile(r"\b(?:show me|tell me|give me|find|create|build|summarize|analyze|explain|fix|debug|track|plan)\s+([a-z][a-z'\s]{2,80})"),
]

CANONICAL_INTENT_PATTERNS: dict[str, list[re.Pattern[str]]] = {
    "Set Reminder": [
        re.compile(r"\b(remind|reminder|notify|alert)\b"),
    ],
    "Manage To-Do List": [
        re.compile(r"\b(to[-\s]?do|todo|checklist|task list)\b"),
        re.compile(r"\b(add|create|update|remove|delete)\s+(?:a\s+)?task\b"),
    ],
    "Schedule Calendar Event": [
        re.compile(r"\b(schedule|reschedule|calendar|appointment|meeting|event)\b"),
    ],
    "Send Email": [
        re.compile(r"\b(send|draft|write)\b.*\b(email|mail|gmail)\b"),
        re.compile(r"\b(email|mail|gmail)\b"),
    ],
    "Send WhatsApp Message": [
        re.compile(r"\b(send|draft|write)\b.*\bwhatsapp\b"),
        re.compile(r"\bwhatsapp\b"),
    ],
    "Create Note Or Summary": [
        re.compile(r"\b(note|notes|summarize|summary|minutes)\b"),
    ],
    "Search / Information Lookup": [
        re.compile(r"\b(search|find|lookup|look up|what is|how to|why)\b"),
    ],
    "Translate Text": [
        re.compile(r"\b(translate|translation)\b"),
    ],
}

FEATURE_FOCUS_PATTERNS: dict[str, list[re.Pattern[str]]] = {
    "Reminders": [
        re.compile(r"\bremind(?:er| me)?\b"),
        re.compile(r"\bremember\b"),
        re.compile(r"\balert\b"),
        re.compile(r"\bnotify\b"),
    ],
    "To-Do Lists": [
        re.compile(r"\bto[-\s]?do\b"),
        re.compile(r"\btask(?:s)?\b"),
        re.compile(r"\bchecklist\b"),
    ],
    "Calendar & Scheduling": [
        re.compile(r"\bcalendar\b"),
        re.compile(r"\bschedule\b"),
        re.compile(r"\bappointment\b"),
        re.compile(r"\bmeeting\b"),
        re.compile(r"\bevent\b"),
        re.compile(r"\bplan\b"),
        re.compile(r"\breschedule\b"),
    ],
    "Notes & Summaries": [
        re.compile(r"\bnote(?:s)?\b"),
        re.compile(r"\bsummar(?:ize|y)\b"),
        re.compile(r"\bminutes\b"),
        re.compile(r"\bjournal\b"),
    ],
    "Search & Q&A": [
        re.compile(r"\bsearch\b"),
        re.compile(r"\bfind\b"),
        re.compile(r"\blookup\b"),
        re.compile(r"\bwhat\b"),
        re.compile(r"\bhow\b"),
        re.compile(r"\bwhy\b"),
    ],
    "Writing & Drafting": [
        re.compile(r"\bwrite\b"),
        re.compile(r"\bdraft\b"),
        re.compile(r"\bemail\b"),
        re.compile(r"\bmessage\b"),
        re.compile(r"\breply\b"),
    ],
    "Translation": [
        re.compile(r"\btranslate\b"),
        re.compile(r"\btranslation\b"),
    ],
}
def heuristic_sentiment_fallback(text: str) -> tuple[float, float]:
    s = str(text or "").strip().lower()
    if not s:
        return 0.0, 0.0

    neg_terms = {
        "bad", "worse", "worst", "hate", "angry", "upset", "frustrated", "annoyed", "disappointed",
        "terrible", "awful", "slow", "broken", "error", "issue", "problem", "failed", "failure",
        "crash", "crashed", "unusable", "useless", "stuck", "bug", "bugs", "lag", "laggy", "refund",
        "pathetic", "horrible", "disaster", "wrong", "inaccurate", "confusing",
    }
    pos_terms = {
        "good", "great", "awesome", "nice", "love", "happy", "thanks", "thankyou", "thank",
        "resolved", "perfect", "excellent", "fast", "smooth", "helpful", "stable", "amazing",
        "works", "working", "fixed", "clear", "accurate",
    }
    neg_phrases = {
        "not working", "doesn't work", "doesnt work", "not useful", "not helpful",
        "not good", "too slow", "very slow", "still broken", "keeps crashing",
        "waste of time", "not satisfied", "not happy", "not accurate", "bad experience",
    }
    pos_phrases = {
        "works well", "very helpful", "super helpful", "thank you", "thanks a lot",
        "well done", "good job", "works great",
    }
    amplifiers = {"very", "really", "extremely", "so", "too", "highly", "super"}
    downtoners = {"slightly", "somewhat", "kinda", "kindof", "kind", "bit", "little"}
    negators = {"not", "never", "no", "none", "hardly", "rarely", "without"}
    contrast_words = {"but", "however", "though", "although", "yet"}

    tokens = re.findall(r"[a-z']+", s)
    if not tokens:
        return 0.0, 0.0

    phrase_neg = sum(1 for p in neg_phrases if p in s)
    phrase_pos = sum(1 for p in pos_phrases if p in s)

    score = 0.0
    last_contrast_idx = max((i for i, t in enumerate(tokens) if t in contrast_words), default=-1)
    for i, tok in enumerate(tokens):
        base = 0.0
        if tok in neg_terms:
            base = -1.0
        elif tok in pos_terms:
            base = 1.0
        if base == 0.0:
            continue

        window = tokens[max(0, i - 3):i]
        amp = 1.0
        if any(w in amplifiers for w in window):
            amp *= 1.35
        if any(w in downtoners for w in window):
            amp *= 0.7
        if any(w in negators for w in window):
            base *= -0.9

        # In "..., but ...", sentiment after the contrast usually carries more intent.
        if last_contrast_idx >= 0 and i > last_contrast_idx:
            amp *= 1.25

        score += base * amp

    score += (1.8 * phrase_pos) - (2.2 * phrase_neg)

    exclam = s.count("!")
    if exclam > 0:
        score *= 1.0 + min(exclam, 3) * 0.08
    if "?" in s and any(w in s for w in ["why", "what", "how", "wtf"]):
        score -= 0.15

    polarity = score / max(len(tokens) ** 0.6, 3.0)
    polarity = float(max(min(polarity, 1.0), -1.0))
    subjectivity = float(min(max(0.2 + 0.8 * abs(polarity), 0.0), 1.0))
    return round(polarity, 4), round(subjectivity, 4)


def style_app() -> None:
    st.set_page_config(page_title="Maya Behavioral Intelligence", page_icon="🔱", layout="wide", initial_sidebar_state="collapsed")
    st.markdown(
        """
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;600;700&family=Inter:wght@300;400;500;600&display=swap');

        /* ═══════════════════════════════════════════════
           ROOT VARIABLES
           ═══════════════════════════════════════════════ */
        :root {
            --bg-primary: #0b1120;
            --bg-secondary: #111827;
            --bg-card: rgba(17, 24, 39, 0.7);
            --border-subtle: rgba(212, 175, 55, 0.12);
            --border-hover: rgba(212, 175, 55, 0.4);
            --accent-gold: #d4af37;
            --accent-gold-dim: rgba(212, 175, 55, 0.15);
            --text-primary: #f1f5f9;
            --text-secondary: #94a3b8;
            --text-muted: #64748b;
            --radius-lg: 16px;
            --radius-md: 12px;
            --radius-sm: 8px;
        }

        /* ═══════════════════════════════════════════════
           GLOBAL RESETS
           ═══════════════════════════════════════════════ */
        [data-testid="stAppViewContainer"] {
            background: linear-gradient(170deg, #0b1120 0%, #111827 40%, #0f172a 100%);
            color: var(--text-primary);
        }
        [data-testid="stAppViewContainer"]::before {
            content: '';
            position: fixed;
            top: 0; left: 0; right: 0; bottom: 0;
            background:
                radial-gradient(ellipse 800px 600px at 10% 15%, rgba(212,175,55,0.04) 0%, transparent 60%),
                radial-gradient(ellipse 600px 400px at 90% 80%, rgba(59,130,246,0.03) 0%, transparent 60%);
            pointer-events: none;
            z-index: 0;
        }
        .main .block-container {
            position: relative;
            z-index: 1;
        }

        /* ═══════════════════════════════════════════════
           HEADER & NAVIGATION
           ═══════════════════════════════════════════════ */
        [data-testid="stSidebarNav"] { display: none; }
        [data-testid="stHeader"] { background: transparent !important; }
        footer { visibility: hidden; }

        /* Sidebar — Frosted Glass */
        [data-testid="stSidebar"] {
            background: rgba(11, 17, 32, 0.92) !important;
            backdrop-filter: blur(20px) saturate(1.8);
            border-right: 1px solid var(--border-subtle);
        }
        [data-testid="stSidebar"] [data-testid="stMarkdownContainer"] p {
            color: var(--text-secondary);
        }

        /* Sidebar Radio Buttons — Pill Navigation */
        [data-testid="stSidebar"] .stRadio > div {
            gap: 4px !important;
        }
        [data-testid="stSidebar"] .stRadio label {
            background: transparent;
            border: 1px solid transparent;
            border-radius: var(--radius-sm);
            padding: 10px 16px !important;
            margin: 0 !important;
            transition: all 0.25s cubic-bezier(0.4, 0, 0.2, 1);
            cursor: pointer;
        }
        [data-testid="stSidebar"] .stRadio label:hover {
            background: var(--accent-gold-dim);
            border-color: var(--border-subtle);
        }
        [data-testid="stSidebar"] .stRadio label[data-checked="true"],
        [data-testid="stSidebar"] .stRadio label:has(input:checked) {
            background: linear-gradient(135deg, rgba(212,175,55,0.15) 0%, rgba(212,175,55,0.05) 100%);
            border-color: var(--accent-gold) !important;
            box-shadow: 0 0 20px rgba(212,175,55,0.08);
        }

        /* Sidebar Toggle Button */
        button[kind="header"] {
            color: var(--accent-gold) !important;
            background: rgba(17, 24, 39, 0.6) !important;
            border-radius: var(--radius-sm) !important;
            border: 1px solid var(--border-subtle) !important;
            transition: all 0.25s ease !important;
        }
        button[kind="header"]:hover {
            border-color: var(--accent-gold) !important;
            background: rgba(17, 24, 39, 0.9) !important;
            box-shadow: 0 0 15px rgba(212,175,55,0.12) !important;
        }

        /* ═══════════════════════════════════════════════
           TYPOGRAPHY
           ═══════════════════════════════════════════════ */
        .block-container {
            padding-top: 1.5rem !important;
            padding-bottom: 2rem;
            font-family: 'Inter', sans-serif;
            max-width: 96% !important;
        }
        h1, h2, h3 {
            font-family: 'Outfit', sans-serif !important;
            color: var(--text-primary) !important;
            font-weight: 700 !important;
        }
        h1 {
            font-size: 2.4rem !important;
            letter-spacing: -0.5px;
            background: linear-gradient(135deg, #f1f5f9 0%, #d4af37 100%);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            background-clip: text;
        }
        h2 { font-size: 1.6rem !important; }
        h3 { font-size: 1.25rem !important; color: #e2e8f0 !important; }

        /* Captions */
        [data-testid="stCaptionContainer"] {
            color: var(--text-muted) !important;
            font-size: 0.82rem !important;
        }

        /* ═══════════════════════════════════════════════
           METRIC CARDS — Glassmorphism
           ═══════════════════════════════════════════════ */
        [data-testid="metric-container"] {
            background: var(--bg-card);
            backdrop-filter: blur(12px);
            border: 1px solid var(--border-subtle);
            border-radius: var(--radius-md);
            padding: 1.2rem;
            box-shadow: 0 4px 24px rgba(0, 0, 0, 0.25), inset 0 1px 0 rgba(255,255,255,0.03);
            transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
        }
        [data-testid="metric-container"]:hover {
            border-color: var(--border-hover);
            box-shadow: 0 8px 32px rgba(212, 175, 55, 0.1), inset 0 1px 0 rgba(255,255,255,0.05);
            transform: translateY(-2px);
        }
        [data-testid="stMetricLabel"] {
            color: var(--text-secondary) !important;
            font-family: 'Inter', sans-serif;
            font-weight: 500;
            letter-spacing: 0.5px;
            text-transform: uppercase;
            font-size: 0.72rem !important;
        }
        [data-testid="stMetricValue"] {
            color: var(--text-primary) !important;
            font-family: 'Outfit', sans-serif;
            font-weight: 700;
        }
        [data-testid="stMetricDelta"] {
            font-family: 'Inter', sans-serif;
            font-weight: 600;
        }

        /* ═══════════════════════════════════════════════
           PLOTLY CHARTS — Dark Glass
           ═══════════════════════════════════════════════ */
        .stPlotlyChart {
            background: rgba(17, 24, 39, 0.5);
            border-radius: var(--radius-lg);
            border: 1px solid rgba(255, 255, 255, 0.04);
            padding: 1rem;
            box-shadow: 0 8px 32px rgba(0, 0, 0, 0.2);
            transition: border-color 0.3s ease;
        }
        .stPlotlyChart:hover {
            border-color: var(--border-subtle);
        }

        /* ═══════════════════════════════════════════════
           TABLES & DATAFRAMES
           ═══════════════════════════════════════════════ */
        .stDataFrame {
            background: rgba(11, 17, 32, 0.6);
            border: 1px solid var(--border-subtle);
            border-radius: var(--radius-md);
            overflow: hidden;
        }
        .stDataFrame [data-testid="glideDataEditor"] {
            border-radius: var(--radius-md);
        }
        /* Header row styling */
        .stDataFrame th, .stDataFrame [role="columnheader"] {
            background: rgba(212, 175, 55, 0.08) !important;
            color: var(--text-secondary) !important;
            font-family: 'Inter', sans-serif;
            font-weight: 600;
            text-transform: uppercase;
            font-size: 0.72rem;
            letter-spacing: 0.5px;
        }

        /* ═══════════════════════════════════════════════
           BUTTONS — Gold Finish
           ═══════════════════════════════════════════════ */
        .stButton > button {
            background: linear-gradient(135deg, #d4af37 0%, #b8962e 100%);
            color: #0b1120 !important;
            border: none;
            padding: 0.65rem 2rem;
            border-radius: var(--radius-sm);
            font-weight: 600;
            font-family: 'Outfit', sans-serif;
            font-size: 0.88rem;
            box-shadow: 0 4px 15px rgba(212, 175, 55, 0.25);
            transition: all 0.25s cubic-bezier(0.4, 0, 0.2, 1);
            letter-spacing: 0.3px;
        }
        .stButton > button:hover {
            box-shadow: 0 8px 25px rgba(212, 175, 55, 0.4);
            transform: translateY(-1px) scale(1.01);
        }
        .stButton > button:active {
            transform: translateY(0) scale(0.99);
        }

        /* Download Button */
        .stDownloadButton > button {
            background: transparent;
            color: var(--accent-gold) !important;
            border: 1px solid var(--border-hover);
            font-family: 'Inter', sans-serif;
            font-weight: 500;
        }
        .stDownloadButton > button:hover {
            background: var(--accent-gold-dim);
        }

        /* ═══════════════════════════════════════════════
           SELECT BOXES & INPUTS
           ═══════════════════════════════════════════════ */
        .stSelectbox > div > div,
        .stMultiSelect > div > div,
        .stTextInput > div > div > input {
            background: var(--bg-secondary) !important;
            border: 1px solid var(--border-subtle) !important;
            border-radius: var(--radius-sm) !important;
            color: var(--text-primary) !important;
            font-family: 'Inter', sans-serif;
            transition: border-color 0.2s ease;
        }
        .stSelectbox > div > div:focus-within,
        .stTextInput > div > div > input:focus {
            border-color: var(--accent-gold) !important;
            box-shadow: 0 0 0 2px rgba(212,175,55,0.1) !important;
        }

        /* ═══════════════════════════════════════════════
           EXPANDERS
           ═══════════════════════════════════════════════ */
        .streamlit-expanderHeader {
            background: var(--bg-card) !important;
            border: 1px solid var(--border-subtle) !important;
            border-radius: var(--radius-sm) !important;
            color: var(--text-primary) !important;
            font-family: 'Outfit', sans-serif;
            font-weight: 600;
        }
        .streamlit-expanderContent {
            border: 1px solid var(--border-subtle) !important;
            border-top: none !important;
            border-radius: 0 0 var(--radius-sm) var(--radius-sm) !important;
            background: rgba(11, 17, 32, 0.4) !important;
        }

        /* ═══════════════════════════════════════════════
           DIVIDERS — Gold Gradient
           ═══════════════════════════════════════════════ */
        [data-testid="stHorizontalBlock"] + hr,
        hr {
            border: none !important;
            height: 1px !important;
            background: linear-gradient(90deg, transparent 0%, rgba(212,175,55,0.3) 50%, transparent 100%) !important;
            margin: 1.5rem 0 !important;
        }

        /* ═══════════════════════════════════════════════
           ALERTS & INFO BOXES
           ═══════════════════════════════════════════════ */
        .stAlert {
            border-radius: var(--radius-md) !important;
            border-left-width: 4px !important;
            backdrop-filter: blur(8px);
        }

        /* ═══════════════════════════════════════════════
           TABS
           ═══════════════════════════════════════════════ */
        .stTabs [data-baseweb="tab-list"] {
            gap: 2px;
            background: rgba(11, 17, 32, 0.5);
            border-radius: var(--radius-sm);
            padding: 4px;
        }
        .stTabs [data-baseweb="tab"] {
            border-radius: 6px;
            color: var(--text-secondary);
            font-family: 'Inter', sans-serif;
            font-weight: 500;
            padding: 8px 20px;
            transition: all 0.2s ease;
        }
        .stTabs [data-baseweb="tab"]:hover {
            color: var(--text-primary);
            background: rgba(212,175,55,0.06);
        }
        .stTabs [aria-selected="true"] {
            color: var(--accent-gold) !important;
            background: rgba(212,175,55,0.1) !important;
        }
        .stTabs [data-baseweb="tab-highlight"] {
            background-color: var(--accent-gold) !important;
        }

        /* ═══════════════════════════════════════════════
           SCROLLBAR
           ═══════════════════════════════════════════════ */
        ::-webkit-scrollbar { width: 6px; height: 6px; }
        ::-webkit-scrollbar-track { background: var(--bg-primary); }
        ::-webkit-scrollbar-thumb {
            background: rgba(212, 175, 55, 0.25);
            border-radius: 3px;
        }
        ::-webkit-scrollbar-thumb:hover { background: rgba(212, 175, 55, 0.45); }

        /* ═══════════════════════════════════════════════
           SPINNER
           ═══════════════════════════════════════════════ */
        .stSpinner > div > div {
            border-top-color: var(--accent-gold) !important;
        }

        /* ═══════════════════════════════════════════════
           TOAST NOTIFICATIONS
           ═══════════════════════════════════════════════ */
        [data-testid="stToast"] {
            background: var(--bg-secondary) !important;
            border: 1px solid var(--border-subtle) !important;
            border-radius: var(--radius-md) !important;
            color: var(--text-primary) !important;
        }

        </style>
        """,
        unsafe_allow_html=True,
    )


def executive_card(label: str, content: str = ""):
    """Renders a premium gold-bordered card for executive summaries."""
    st.markdown(
        f"""
        <div style="
            background: rgba(17, 24, 39, 0.65);
            backdrop-filter: blur(16px) saturate(1.4);
            border: 1px solid rgba(212, 175, 55, 0.2);
            border-radius: 14px;
            padding: 1.5rem;
            margin-bottom: 1rem;
            box-shadow: 0 8px 32px rgba(0,0,0,0.35), inset 0 1px 0 rgba(255,255,255,0.03);
            transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
        ">
            <div style="font-family: 'Inter', sans-serif; color: #94a3b8; font-size: 0.78rem; text-transform: uppercase; letter-spacing: 1.2px; margin-bottom: 0.5rem;">
                {label}
            </div>
            <div style="font-family: 'Outfit', sans-serif; color: #f1f5f9; font-size: 1.8rem; font-weight: 700;">
                {content}
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def executive_metric(label: str, value: str, delta: str = ""):
    """Custom metric component for high-end data cards."""
    color = "#2ECC71" if "+" in str(delta) else "#E74C3C"
    delta_html = f'<span style="color: {color}; font-size: 0.85rem; margin-left: 8px; font-weight: 600;">{delta}</span>' if delta else ""
    
    st.markdown(f"""
        <div style="
            background: rgba(17, 24, 39, 0.5);
            border-left: 3px solid rgba(212, 175, 55, 0.7);
            border-radius: 4px 12px 12px 4px;
            padding: 1rem 1.2rem;
            margin-bottom: 0.5rem;
            transition: all 0.25s cubic-bezier(0.4, 0, 0.2, 1);
            box-shadow: 0 2px 8px rgba(0,0,0,0.15);
        ">
            <div style="color: #94a3b8; font-size: 0.72rem; text-transform: uppercase; letter-spacing: 0.8px; margin-bottom: 5px; font-family: 'Inter', sans-serif;">{label}</div>
            <div style="color: #f1f5f9; font-size: 1.5rem; font-weight: 700; font-family: 'Outfit', sans-serif;">
                {value}{delta_html}
            </div>
        </div>
    """, unsafe_allow_html=True)


def style_chart(
    fig,
    height: int = 460,
    x_title: str | None = None,
    y_title: str | None = None,
    rotate_x: bool = False,
    kind: str = "cartesian",
):
    # Ensure layout exists and is clean
    fig.update_layout(
        height=height,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(size=12, color="#F1F5F9", family="Inter"),
        margin=dict(l=50, r=20, t=50, b=50),
        showlegend=True,
    )
    
    # Explicitly clear title to prevent 'undefined' rendering
    if not (hasattr(fig.layout, 'title') and fig.layout.title.text):
        fig.update_layout(title_text="")
    else:
        fig.update_layout(title_font=dict(size=18, color="#F8FAFC", family="Outfit"))

    if kind == "pie":
        fig.update_traces(
            textposition="outside",
            textinfo="percent+label",
            marker=dict(line=dict(color="#0f172a", width=2)),
        )
        return fig

    # Cartesian-only updates
    fig.update_xaxes(gridcolor="rgba(255,255,255,0.05)", zeroline=False)
    fig.update_yaxes(gridcolor="rgba(255,255,255,0.05)", zeroline=False)

    if kind == "cartesian":
        fig.update_xaxes(
            title=x_title if x_title else None,
            tickangle=-25 if rotate_x else 0,
            tickfont=dict(size=11, color="#CBD5E1"),
        )
        fig.update_yaxes(
            title=y_title if y_title else None,
            tickfont=dict(size=11, color="#CBD5E1"),
        )

        for tr in fig.data:
            if getattr(tr, "type", "") in {"bar", "histogram", "scatter", "scattergl"}:
                tr.update(marker_line_width=1)
        if len(fig.data) == 1 and getattr(fig.data[0], "type", "") in {"bar", "histogram", "scatter"}:
            fig.update_traces(marker_color=ACCENT_PRIMARY)
    return fig


def remove_geographic_noise(df: pd.DataFrame, feature_col: str = "feature") -> pd.DataFrame:
    if df.empty or feature_col not in df.columns:
        return df
    keep_mask = ~df[feature_col].astype(str).str.contains(GEO_NOISE_PATTERN, case=False, regex=True)
    return df[keep_mask].copy()


def remove_non_actionable_feature_noise(df: pd.DataFrame, feature_col: str = "feature") -> pd.DataFrame:
    if df.empty or feature_col not in df.columns:
        return df
    # Filter out blocked keys and any categorical types that don't add value to interpretation
    blocked_features = {"type_customer"}
    mask = ~df[feature_col].astype(str).str.lower().str.strip().isin(blocked_features)
    df = df[mask].copy()
    
    # Also remove any feature starting with 'type_' as requested by user (User Type bars)
    mask_type = ~df[feature_col].astype(str).str.lower().str.startswith("type_")
    return df[mask_type].copy()


def humanize_feature_name(name: str) -> str:
    raw = str(name).strip()
    if not raw:
        return ""

    key = raw.lower()
    if key.startswith("type_"):
        return f"User Type: {raw.split('_', 1)[1].replace('_', ' ').title()}"
    if key.startswith("status_"):
        return f"Account Status: {raw.split('_', 1)[1].replace('_', ' ').title()}"
    if key.startswith("country_"):
        v = raw.split("_", 1)[1].replace("_", " ")
        return "Country Missing" if v.lower() == "unknown" else f"Country: {v.upper() if len(v) <= 3 else v.title()}"
    if key.startswith("timezone_"):
        tz = raw.split("_", 1)[1].replace("_", " ").replace("/", " / ")
        return "Timezone Missing" if tz.strip().lower() == "unknown" else f"Timezone: {tz}"

    s = re.sub(r"(?<=[a-z0-9])(?=[A-Z])", " ", raw)
    s = s.replace("_", " ")
    s = re.sub(r"\s+", " ", s).strip()

    word_map = {
        "avg": "Average",
        "mean": "Average",
        "msg": "Message",
        "msgs": "Messages",
        "cnt": "Count",
        "num": "Number",
        "std": "Std Dev",
        "min": "Minimum",
        "max": "Maximum",
        "emb": "Embedding",
        "len": "Length",
    }
    upper_map = {"id": "ID", "utc": "UTC", "gnn": "GNN", "shap": "SHAP", "tsne": "t-SNE"}

    out: list[str] = []
    for w in s.split(" "):
        lw = w.lower()
        if lw in upper_map:
            out.append(upper_map[lw])
        elif lw in word_map:
            out.append(word_map[lw])
        elif lw.isdigit():
            out.append(lw)
        else:
            out.append(lw.capitalize())

    return " ".join(out)


def shorten_user_label(label: str, max_chars: int = 24) -> str:
    s = str(label)
    return s if len(s) <= max_chars else s[: max_chars - 1] + "…"


def simplify_persona_label(label: str) -> str:
    s = str(label or "").strip()
    if not s:
        return s

    # Convert technical persona wording to plain language while preserving [P#] suffixes.
    replacements = [
        ("Frustrated", "Needs Help"),
        ("Satisfied", "Happy"),
        ("Neutral", "Steady"),
        ("Long-term", "Long-Time"),
        ("Highly Active", "Very Active"),
        ("Low Activity", "Less Active"),
    ]

    out = s
    for old, new in replacements:
        out = re.sub(rf"\\b{re.escape(old)}\\b", new, out, flags=re.IGNORECASE)

    out = re.sub(r"\\bUsers\\b", "", out, flags=re.IGNORECASE)
    out = re.sub(r"\\s{2,}", " ", out).strip()
    return out


@st.cache_resource(show_spinner=False)
def get_redis_client():
    redis_url = os.getenv("REDIS_URL", "").strip()
    if not redis_url:
        return None

    try:
        import redis  # type: ignore
    except Exception:
        return None

    try:
        timeout = float(os.getenv("MAYA_REDIS_TIMEOUT_SEC", "0.5"))
        client = redis.Redis.from_url(
            redis_url,
            decode_responses=True,
            socket_connect_timeout=timeout,
            socket_timeout=timeout,
            retry_on_timeout=False,
        )
        client.ping()
        return client
    except Exception:
        return None


def _empty_df(columns: list[str] | None = None) -> pd.DataFrame:
    return pd.DataFrame(columns=columns or [])


def load_df_from_redis(key: str, expected_cols: list[str] | None = None) -> pd.DataFrame:
    client = get_redis_client()
    if client is None:
        return _empty_df(expected_cols)

    redis_key = f"{REDIS_KEY_PREFIX}:{key}"
    try:
        payload = client.get(redis_key)
    except Exception:
        return _empty_df(expected_cols)

    if not payload:
        return _empty_df(expected_cols)

    try:
        records = json.loads(payload)
        df = pd.DataFrame(records)
    except Exception:
        return _empty_df(expected_cols)

    if expected_cols:
        for c in expected_cols:
            if c not in df.columns:
                df[c] = np.nan
        return df[expected_cols + [c for c in df.columns if c not in expected_cols]]
    return df


def get_dashboard_last_updated_label() -> str:
    client = get_redis_client()
    if client is not None:
        try:
            payload = client.get(f"{REDIS_KEY_PREFIX}:last_published_at")
            if payload:
                ts = pd.to_datetime(payload, utc=True, errors="coerce")
                if pd.notna(ts):
                    local_ts = ts.tz_convert("Asia/Kolkata")
                    return local_ts.strftime("%Y-%m-%d %H:%M:%S %Z")
        except Exception:
            pass

    tracked_files = [
        OUTPUT_DIR / "user_behaviour_scores.csv",
        OUTPUT_DIR / "user_feature_importance_global.csv",
        OUTPUT_DIR / "user_feature_importance_per_user.csv",
        USER_EMBEDDINGS_PATH,
        XGB_PREDICTIONS_PATH,
        PERSONA_TABLE_PATH,
        SENTIMENT_SCORES_PATH,
        GRU_MOOD_SWING_SUMMARY_PATH,
    ]
    existing = [p for p in tracked_files if p.exists()]
    if not existing:
        return "Unavailable"

    latest = max(existing, key=lambda p: p.stat().st_mtime)
    dt = datetime.fromtimestamp(latest.stat().st_mtime)
    return dt.strftime("%Y-%m-%d %H:%M:%S")


def get_data_refresh_nonce() -> str:
    client = get_redis_client()
    if client is not None:
        try:
            payload = client.get(f"{REDIS_KEY_PREFIX}:last_published_at")
            if payload:
                return str(payload)
        except Exception:
            pass

    tracked_files = [
        OUTPUT_DIR / "user_behaviour_scores.csv",
        OUTPUT_DIR / "user_feature_importance_global.csv",
        OUTPUT_DIR / "user_feature_importance_per_user.csv",
        SENTIMENT_SCORES_PATH,
    ]
    existing = [p for p in tracked_files if p.exists()]
    if not existing:
        return "no-data"
    latest = max(existing, key=lambda p: p.stat().st_mtime)
    return str(int(latest.stat().st_mtime))
