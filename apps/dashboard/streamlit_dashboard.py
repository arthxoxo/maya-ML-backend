"""
Simple frontend dashboard for:
1) User-based feature importance (from trained GNN outputs)
2) User-based sentiment analysis (from gnn_preprocessed message nodes)

Run:
    python -m streamlit run streamlit_dashboard.py
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

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import plotly.io as pio
import streamlit as st
from sklearn.manifold import TSNE

# Ensure project-root imports (e.g., app_config) resolve even when launching from nested dirs.
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import app_config as cfg

BASE_DIR = cfg.BASE_DIR
GNN_OUTPUT_DIR = cfg.GNN_OUTPUT_DIR
GNN_PREPROCESSED_DIR = cfg.GNN_PREPROCESSED_DIR
SECRET_DATA_DIR = cfg.SECRET_DATA_DIR
FLINK_ENGINEERED_DIR = getattr(cfg, "FLINK_ENGINEERED_DIR", BASE_DIR / "flink_engineered")

# Backward-compatible path resolution for environments where config naming drifted.
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

SENTIMENT_COLORS = {
    "positive": "#1F7A5A",
    "neutral": "#5F6B7A",
    "negative": "#B2413E",
}

SENTIMENT_DIVERGING_SCALE = [
    [0.00, "#9F2D2D"],
    [0.35, "#D57452"],
    [0.50, "#F6F3EE"],
    [0.65, "#79AF8E"],
    [1.00, "#1F7A5A"],
]

ACCENT_PRIMARY = "#8D6A3B"
CHART_PAPER_BG = "#F8F1E4"
CHART_PLOT_BG = "#FFFDF8"
GRID_COLOR = "#E5D8C0"
PERSONA_COLORS = ["#8D6A3B", "#1E3A5F", "#A46B4D", "#556B5D", "#6B4C3E", "#35606E", "#A1834C"]
RISK_COLORS = {"High": "#B2413E", "Medium": "#D29A2E", "Low": "#2E8B57"}
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
    st.set_page_config(page_title="Maya GNN Insights", page_icon="M", layout="wide")
    st.markdown(
        """
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Cormorant+Garamond:wght@500;600;700&family=Manrope:wght@400;500;600;700&display=swap');
        [data-testid="stAppViewContainer"] {
            background:
                radial-gradient(circle at 8% 12%, rgba(141,106,59,0.16), transparent 36%),
                radial-gradient(circle at 92% 84%, rgba(30,58,95,0.12), transparent 34%),
                linear-gradient(180deg, #f9f4ea 0%, #f4ece0 100%);
        }
        [data-testid="stHeader"] {
            background: rgba(249,244,234,0.70);
            backdrop-filter: blur(6px);
        }
        [data-testid="stSidebar"] {
            background: linear-gradient(180deg, #1f304a 0%, #243b5a 42%, #2f4566 100%);
            border-right: 1px solid rgba(255,255,255,0.12);
        }
        [data-testid="stSidebar"] * {
            color: #f6efe3 !important;
        }
        [data-testid="stSidebar"] .stSelectbox label,
        [data-testid="stSidebar"] .stRadio label {
            color: #f2e5cb !important;
            font-family: "Manrope", "Segoe UI", sans-serif;
            font-weight: 600;
        }
        .block-container {
            font-family: "Manrope", "Segoe UI", sans-serif;
            color: #1f2125;
            padding-top: 1.3rem;
            padding-bottom: 2.5rem;
        }
        h1, h2, h3 {
            font-family: "Cormorant Garamond", "Times New Roman", serif;
            letter-spacing: 0.25px;
            color: #1c2737;
        }
        h1 {
            font-weight: 700;
            font-size: 3.0rem;
        }
        h2, h3 {
            font-weight: 600;
        }
        .stPlotlyChart {
            border: 1px solid #dccfb5;
            border-radius: 16px;
            padding: 0.55rem 0.55rem 0.15rem 0.55rem;
            background: linear-gradient(180deg, #fffdf9 0%, #f7efe1 100%);
            box-shadow: 0 10px 28px rgba(35, 32, 28, 0.09);
            transition: transform 0.2s ease, box-shadow 0.2s ease;
        }
        .stPlotlyChart:hover {
            transform: translateY(-1px);
            box-shadow: 0 14px 34px rgba(35, 32, 28, 0.13);
        }
        [data-testid="stMetric"] {
            background: linear-gradient(160deg, rgba(255,255,255,0.88), rgba(247,238,224,0.95));
            border: 1px solid #ddcfb3;
            border-radius: 14px;
            padding: 0.8rem 0.9rem;
            box-shadow: 0 6px 20px rgba(52, 45, 33, 0.09);
        }
        [data-testid="stMetricLabel"] {
            font-family: "Manrope", "Segoe UI", sans-serif;
            color: #5d4d37;
            font-weight: 600;
        }
        [data-testid="stMetricValue"] {
            font-family: "Cormorant Garamond", "Times New Roman", serif;
            color: #1f2733;
            font-size: 2rem;
        }
        .stDataFrame {
            border: 1px solid #decfae;
            border-radius: 14px;
            box-shadow: 0 6px 18px rgba(52,45,33,0.09);
            overflow: hidden;
        }
        .stButton > button {
            background: linear-gradient(135deg, #8d6a3b, #b58d55);
            color: #fffaf0;
            border: 1px solid #7d5e35;
            border-radius: 10px;
            font-weight: 600;
            box-shadow: 0 6px 14px rgba(83, 58, 24, 0.25);
        }
        .stButton > button:hover {
            filter: brightness(1.03);
        }
        </style>
        """,
        unsafe_allow_html=True,
    )

    maya_template = go.layout.Template(
        layout=go.Layout(
            font=dict(family="Manrope, Segoe UI, sans-serif", size=14, color="#1e2229"),
            title=dict(font=dict(size=22, color="#1c2737")),
            paper_bgcolor=CHART_PAPER_BG,
            plot_bgcolor=CHART_PLOT_BG,
            legend=dict(bgcolor="rgba(255,250,240,0.97)", bordercolor="#d8c8aa", borderwidth=1),
            margin=dict(l=64, r=24, t=66, b=58),
            xaxis=dict(showgrid=True, gridcolor=GRID_COLOR, zeroline=False),
            yaxis=dict(showgrid=True, gridcolor=GRID_COLOR, zeroline=False),
        )
    )
    pio.templates["maya_readable"] = maya_template
    pio.templates.default = "maya_readable"
    px.defaults.template = "maya_readable"
    px.defaults.color_discrete_sequence = PERSONA_COLORS


def style_chart(
    fig,
    height: int = 460,
    x_title: str | None = None,
    y_title: str | None = None,
    rotate_x: bool = False,
    kind: str = "cartesian",
):
    fig.update_layout(
        height=height,
        title_font=dict(size=21, color="#1c2737"),
        font=dict(size=13, color="#20242b"),
        legend=dict(yanchor="top", y=1.02, xanchor="left", x=0.01, bgcolor="rgba(255,251,243,0.96)", font=dict(size=12)),
        hoverlabel=dict(font_size=13, bgcolor="#fffaf2", bordercolor="#d9c8a7", font_color="#1e2228"),
        margin=dict(l=62, r=22, t=64, b=58),
        paper_bgcolor=CHART_PAPER_BG,
        plot_bgcolor=CHART_PLOT_BG,
    )

    if kind == "pie":
        fig.update_traces(
            textposition="outside",
            textfont_size=13,
            textinfo="percent+label",
            marker=dict(line=dict(color="#ffffff", width=2)),
            insidetextfont=dict(color="#ffffff", size=12),
            outsidetextfont=dict(color="#1d2129", size=12),
            automargin=True,
        )
        return fig

    if kind == "cartesian":
        fig.update_xaxes(
            title=x_title,
            tickangle=-25 if rotate_x else 0,
            automargin=True,
            showline=True,
            linecolor="#b7a98d",
            tickfont=dict(size=12, color="#22262e"),
            title_font=dict(size=13, color="#283445"),
            gridcolor=GRID_COLOR,
        )
        fig.update_yaxes(
            title=y_title,
            automargin=True,
            showline=True,
            linecolor="#b7a98d",
            gridcolor=GRID_COLOR,
            tickfont=dict(size=12, color="#22262e"),
            title_font=dict(size=13, color="#283445"),
        )

        fig.update_traces(marker_line_color="#ffffff", marker_line_width=1)
        if len(fig.data) == 1 and getattr(fig.data[0], "type", "") in {"bar", "histogram", "scatter"}:
            fig.update_traces(marker_color=ACCENT_PRIMARY)
    return fig


def remove_geographic_noise(df: pd.DataFrame, feature_col: str = "feature") -> pd.DataFrame:
    if df.empty or feature_col not in df.columns:
        return df
    keep_mask = ~df[feature_col].astype(str).str.contains(GEO_NOISE_PATTERN, case=False, regex=True)
    return df[keep_mask].copy()


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


@st.cache_data(show_spinner=False)
def load_outputs() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    scores_r = load_df_from_redis("user_behaviour_scores", expected_cols=["user_id", "engagement_score", "pred_high_engagement_prob"])
    global_r = load_df_from_redis("user_feature_importance_global", expected_cols=["feature", "importance"])
    per_user_r = load_df_from_redis("user_feature_importance_per_user", expected_cols=["user_id", "rank", "feature", "importance"])

    if not scores_r.empty and not global_r.empty and not per_user_r.empty:
        scores_r["user_id"] = pd.to_numeric(scores_r["user_id"], errors="coerce")
        per_user_r["user_id"] = pd.to_numeric(per_user_r["user_id"], errors="coerce")
        per_user_r["rank"] = pd.to_numeric(per_user_r["rank"], errors="coerce")
        global_r["importance"] = pd.to_numeric(global_r["importance"], errors="coerce").fillna(0.0)
        per_user_r["importance"] = pd.to_numeric(per_user_r["importance"], errors="coerce").fillna(0.0)
        return scores_r.dropna(subset=["user_id"]), global_r, per_user_r.dropna(subset=["user_id"])

    scores = pd.read_csv(OUTPUT_DIR / "user_behaviour_scores.csv")
    global_imp = pd.read_csv(OUTPUT_DIR / "user_feature_importance_global.csv")
    per_user_imp = pd.read_csv(OUTPUT_DIR / "user_feature_importance_per_user.csv")
    return scores, global_imp, per_user_imp


@st.cache_data(show_spinner=False)
def load_user_directory() -> pd.DataFrame:
    frames: list[pd.DataFrame] = []

    users_raw = load_df_from_redis("users_nodes", expected_cols=["user_id", "first_name", "last_name", "full_name"])
    if not users_raw.empty:
        frames.append(users_raw.copy())

    users_path = PREPROCESSED_DIR / "users_nodes.csv"
    if users_path.exists():
        frames.append(pd.read_csv(users_path))

    # Strong fallback: raw users table from secret_data.
    if RAW_USERS_PATH.exists():
        raw = pd.read_csv(RAW_USERS_PATH)
        raw = raw.rename(columns={"id": "user_id"})
        frames.append(raw)

    if not frames:
        return pd.DataFrame(columns=["user_id", "display_name"])

    users = pd.concat(frames, ignore_index=True, sort=False)
    users["user_id"] = pd.to_numeric(users.get("user_id"), errors="coerce")
    users = users.dropna(subset=["user_id"]).copy()
    users["user_id"] = users["user_id"].astype(int)
    users["first_name"] = users.get("first_name", "").fillna("").astype(str).str.strip()
    users["last_name"] = users.get("last_name", "").fillna("").astype(str).str.strip()
    users["full_name"] = users.get("full_name", "").fillna("").astype(str).str.strip()
    users.loc[users["full_name"].eq(""), "full_name"] = (users["first_name"] + " " + users["last_name"]).str.strip()
    users = users.sort_values(["user_id", "full_name"], ascending=[True, False]).drop_duplicates("user_id", keep="first")
    users["display_name"] = users["full_name"].replace("", np.nan).fillna("User") + " (" + users["user_id"].astype(str) + ")"
    return users[["user_id", "display_name"]].drop_duplicates("user_id")


@st.cache_data(show_spinner=False)
def load_user_profiles() -> pd.DataFrame:
    users_r = load_df_from_redis(
        "users_nodes",
        expected_cols=["user_id", "created_at", "timezone", "country", "city", "state"],
    )
    if not users_r.empty:
        users = users_r.copy()
    else:
        users_path = PREPROCESSED_DIR / "users_nodes.csv"
        if not users_path.exists():
            return pd.DataFrame(columns=["user_id", "created_at", "timezone", "country", "city", "state"])
        users = pd.read_csv(users_path)

    users["user_id"] = pd.to_numeric(users.get("user_id"), errors="coerce")
    users = users.dropna(subset=["user_id"]).copy()
    users["user_id"] = users["user_id"].astype(int)
    users["created_at"] = pd.to_datetime(users.get("created_at"), errors="coerce", utc=True)
    for c in ["timezone", "country", "city", "state"]:
        if c in users.columns:
            users[c] = users[c].fillna("").astype(str).str.strip()
        else:
            users[c] = ""
    return users[["user_id", "created_at", "timezone", "country", "city", "state"]].drop_duplicates("user_id", keep="last")


def polarity_label(p: float) -> str:
    if p > 0.08:
        return "positive"
    if p < -0.06:
        return "negative"
    return "neutral"


def apply_negative_boost_from_text(
    df: pd.DataFrame,
    text_col: str = "message",
    polarity_col: str = "polarity",
) -> pd.DataFrame:
    if df.empty or text_col not in df.columns or polarity_col not in df.columns:
        return df

    out = df.copy()
    heur = out[text_col].fillna("").astype(str).apply(heuristic_sentiment_fallback)
    heur_pol = pd.to_numeric(heur.str[0], errors="coerce").fillna(0.0)
    base_pol = pd.to_numeric(out[polarity_col], errors="coerce").fillna(0.0)

    # Bias toward stronger negatives from text when baseline is neutral/weakly negative.
    boost_mask = heur_pol < (base_pol - 0.05)
    out[polarity_col] = np.where(boost_mask, 0.65 * base_pol + 0.35 * heur_pol, base_pol)
    out[polarity_col] = pd.to_numeric(out[polarity_col], errors="coerce").fillna(0.0).clip(-1.0, 1.0)
    return out


def repair_flat_sentiment_scores(
    df: pd.DataFrame,
    text_col: str = "message",
    score_col: str = "sentiment_score",
    label_col: str = "sentiment_label",
) -> pd.DataFrame:
    if df.empty:
        return df
    if text_col not in df.columns:
        return df

    out = df.copy()
    score = pd.to_numeric(out.get(score_col), errors="coerce").fillna(0.0)
    label = out.get(label_col, "").fillna("").astype(str).str.lower().str.strip()

    flat_ratio = float((score.abs() <= 1e-9).mean())
    neutral_ratio = float((label == "neutral").mean()) if len(label) else 1.0
    if flat_ratio < 0.80 and neutral_ratio < 0.90:
        return out

    heur = out[text_col].fillna("").astype(str).apply(lambda t: float(heuristic_sentiment_fallback(t)[0]))
    weak_mask = score.abs() < 0.03
    blended = np.where(weak_mask, heur, 0.75 * score + 0.25 * heur)
    out[score_col] = pd.Series(blended, index=out.index, dtype="float64").fillna(0.0).clip(-1.0, 1.0)
    out[label_col] = out[score_col].apply(polarity_label)
    return out


def _read_csv_subset(path: Path, desired_cols: list[str]) -> pd.DataFrame:
    try:
        available = pd.read_csv(path, nrows=0).columns.tolist()
        cols = [c for c in desired_cols if c in available]
        if not cols:
            return pd.DataFrame(columns=desired_cols)
        return pd.read_csv(path, usecols=cols)
    except Exception:
        return pd.read_csv(path)


def _title_from_identifier(text: str) -> str:
    return str(text).replace("_", " ").replace("-", " ").strip().title()


def _derive_city_state(profile_row: pd.Series) -> tuple[str, str]:
    city = str(profile_row.get("city", "")).strip()
    state = str(profile_row.get("state", "")).strip()
    timezone = str(profile_row.get("timezone", "")).strip()

    if city and state:
        return city, state
    if city:
        return city, "Unknown"
    if state:
        return "Unknown", state

    if "/" in timezone:
        tz_tail = timezone.split("/", 1)[1]
        parts = [p for p in re.split(r"[\\/]", tz_tail) if p]
        if parts:
            city_guess = _title_from_identifier(parts[-1])
            state_guess = _title_from_identifier(parts[0]) if len(parts) > 1 else "Unknown"
            return city_guess if city_guess else "Unknown", state_guess if state_guess else "Unknown"
    return "Unknown", "Unknown"


def _sigmoid(x: float) -> float:
    return 1.0 / (1.0 + math.exp(-x))


def _normalize_model_label(lbl: str) -> str:
    l = str(lbl).strip().lower()
    if "positive" in l:
        return "positive"
    if "negative" in l:
        return "negative"
    if "neutral" in l:
        return "neutral"
    # Common fallback for unnamed labels.
    if l in {"label_0", "0"}:
        return "negative"
    if l in {"label_1", "1"}:
        return "neutral"
    if l in {"label_2", "2"}:
        return "positive"
    return "neutral"


def prettify_embedding_feature_name(feature: str) -> str:
    m = re.match(r"^emb_(\d+)$", str(feature).strip(), flags=re.IGNORECASE)
    if m:
        return f"Embedding Dimension {m.group(1)}"
    return str(feature)


def clean_embedding_display_label(label: str) -> str:
    s = str(label).strip()
    # Convert "emb_41 - Session Count" -> "Embedding Dimension 41 - Session Count"
    return re.sub(r"^emb_(\d+)\b", r"Embedding Dimension \1", s, flags=re.IGNORECASE)


def file_updated_caption(path: Path) -> str:
    try:
        ts = pd.Timestamp(path.stat().st_mtime, unit="s", tz="UTC")
        return f"Updated: {ts.tz_convert('Asia/Kolkata').strftime('%Y-%m-%d %H:%M:%S %Z')}"
    except Exception:
        return "Updated: unknown"


def human_signal_name(raw: str) -> str:
    key = str(raw).strip().lower()
    mapping = {
        "session_count": "Session Frequency",
        "session_duration_sum": "Total Session Time",
        "session_duration_mean": "Average Session Length",
        "billed_duration_sum": "Billed Session Time",
        "message_count": "Message Activity",
        "msg_word_len_mean": "Message Length (Words)",
        "msg_char_len_mean": "Message Length (Characters)",
        "input_tokens_sum": "User Input Volume",
        "output_tokens_sum": "Assistant Response Length",
        "cost_usd_sum": "Usage Cost",
        "feedback_count": "Feedback Frequency",
        "feedback_word_len_mean": "Feedback Detail (Words)",
        "feedback_char_len_mean": "Feedback Detail (Characters)",
        "feedback_char_len": "Feedback Detail (Characters)",
        "feedback_word_len": "Feedback Detail (Words)",
        "feedback_avg_sentiment": "Feedback Tone",
        "account_age_days": "Account Tenure",
        "contacts_backfilled": "Contact Sync Status",
        "transcription_ratio": "Voice Interaction Usage",
        "summary_ratio": "Summary Usage",
        "has_summary": "Summary Availability",
        "has_transcription": "Transcription Availability",
    }
    if key in mapping:
        return mapping[key]
    return humanize_feature_name(raw)


def human_signal_explainer(signal_name: str) -> str:
    s = str(signal_name).lower()
    if "session" in s and "frequency" in s:
        return "How often this user comes back."
    if "session length" in s or "total session time" in s:
        return "How long conversations usually last."
    if "message activity" in s:
        return "How actively the user messages."
    if "response length" in s or "input volume" in s:
        return "How verbose the conversations are."
    if "feedback" in s:
        return "How much and how detailed feedback is."
    if "voice interaction" in s or "transcription" in s:
        return "How much the user uses voice/transcribed interactions."
    if "tenure" in s:
        return "How long the account has been active."
    if "cost" in s:
        return "Estimated resource usage for this user."
    return "Behavior signal linked to this embedding dimension."


@st.cache_resource(show_spinner=False)
def load_hf_pipelines():
    try:
        from transformers import pipeline

        sent_pipe = pipeline("sentiment-analysis", model=HF_SENTIMENT_MODEL, tokenizer=HF_SENTIMENT_MODEL, device=-1)
        irony_pipe = pipeline("text-classification", model=HF_IRONY_MODEL, tokenizer=HF_IRONY_MODEL, device=-1)
        return sent_pipe, irony_pipe
    except Exception:
        return None, None


def contextual_hf_sentiment(data: pd.DataFrame, context_window: int = 3) -> pd.DataFrame:
    sent_pipe, irony_pipe = load_hf_pipelines()
    if sent_pipe is None or irony_pipe is None or data.empty:
        return data

    df = data.copy()
    df = df.sort_values(["user_id", "created_at"], kind="mergesort").reset_index(drop=True)

    inference_texts: list[str] = []
    irony_texts: list[str] = []

    for _, grp in df.groupby("user_id", sort=False):
        history: list[str] = []
        for _, row in grp.iterrows():
            msg = str(row.get("message", "")).strip()
            context = " ".join(history[-context_window:]).strip()
            if context:
                inference_texts.append(f"Context: {context} [SEP] Current: {msg}")
            else:
                inference_texts.append(f"Current: {msg}")
            irony_texts.append(msg if msg else " ")

            if msg:
                history.append(msg[:240])

    if not inference_texts:
        return df

    sent_out = sent_pipe(inference_texts, truncation=True, max_length=256, batch_size=24)
    irony_out = irony_pipe(irony_texts, truncation=True, max_length=128, batch_size=24)

    polarities: list[float] = []
    subjectivities: list[float] = []
    final_labels: list[str] = []

    for s_raw, i_raw in zip(sent_out, irony_out):
        s_label = _normalize_model_label(s_raw.get("label", "neutral"))
        s_score = float(s_raw.get("score", 0.0))

        if s_label == "positive":
            base = s_score
        elif s_label == "negative":
            base = -s_score
        else:
            base = 0.0

        irony_label = str(i_raw.get("label", "")).lower()
        irony_score = float(i_raw.get("score", 0.0))
        is_ironic = ("irony" in irony_label and "non" not in irony_label) or irony_label in {"label_1", "1"}

        # If ironic, dampen and partially invert polarity to reduce false-positive literal sentiment.
        if is_ironic and irony_score > 0.60:
            base = -0.45 * base

        # Keep subjectivity-like value from confidence and irony strength.
        subj = float(min(max(0.35 + 0.55 * abs(base) + 0.20 * (irony_score if is_ironic else 0.0), 0.0), 1.0))

        polarities.append(float(max(min(base, 1.0), -1.0)))
        subjectivities.append(subj)
        final_labels.append(polarity_label(base))

    df["polarity"] = polarities
    df["subjectivity"] = subjectivities
    df["sentiment"] = final_labels
    df["sentiment_source"] = "huggingface_contextual"
    return df


@st.cache_data(show_spinner=False)
def load_sentiment_table() -> pd.DataFrame:
    # Fast path: if precomputed sentiment artifact exists, use it to avoid expensive re-inference on page load.
    if SENTIMENT_SCORES_PATH.exists():
        try:
            cached = pd.read_csv(SENTIMENT_SCORES_PATH)
            if not cached.empty:
                if "user_id" in cached.columns:
                    cached["user_id"] = pd.to_numeric(cached["user_id"], errors="coerce")
                    cached = cached.dropna(subset=["user_id"]).copy()
                    cached["user_id"] = cached["user_id"].astype(int)
                if "message" in cached.columns:
                    cached["message"] = cached["message"].fillna("").astype(str).str.strip()
                else:
                    cached["message"] = ""
                if "created_at" in cached.columns:
                    cached["created_at"] = pd.to_datetime(cached["created_at"], errors="coerce", utc=True)
                else:
                    cached["created_at"] = pd.NaT
                if "polarity" not in cached.columns and "sentiment_score" in cached.columns:
                    cached["polarity"] = pd.to_numeric(cached["sentiment_score"], errors="coerce").fillna(0.0)
                cached["polarity"] = pd.to_numeric(cached.get("polarity"), errors="coerce").fillna(0.0)
                cached = apply_negative_boost_from_text(cached, text_col="message", polarity_col="polarity")
                if "subjectivity" not in cached.columns:
                    cached["subjectivity"] = cached["polarity"].abs().clip(0.0, 1.0)
                cached["subjectivity"] = pd.to_numeric(cached.get("subjectivity"), errors="coerce").fillna(0.0)
                if "sentiment" not in cached.columns:
                    if "sentiment_label" in cached.columns:
                        lbl = cached["sentiment_label"].fillna("").astype(str).str.lower().str.strip()
                        cached["sentiment"] = lbl.where(lbl.isin(["positive", "negative", "neutral"]), cached["polarity"].apply(polarity_label))
                    else:
                        cached["sentiment"] = cached["polarity"].apply(polarity_label)
                if "source" not in cached.columns:
                    cached["source"] = "user_message"
                if "sentiment_source" not in cached.columns:
                    cached["sentiment_source"] = "artifact_cache"
                return cached
        except Exception:
            pass

    texts = []

    msg_path = PREPROCESSED_DIR / "messages_nodes.csv"
    sess_path = PREPROCESSED_DIR / "sessions_nodes.csv"
    if msg_path.exists() and sess_path.exists():
        msg = _read_csv_subset(
            msg_path,
            ["session_id", "message_id", "role", "message", "created_at", "sentiment_score", "sentiment_label"],
        )
        sess = _read_csv_subset(sess_path, ["session_id", "user_id"])

        msg["session_id"] = pd.to_numeric(msg.get("session_id"), errors="coerce")
        sess["session_id"] = pd.to_numeric(sess.get("session_id"), errors="coerce")
        sess["user_id"] = pd.to_numeric(sess.get("user_id"), errors="coerce")

        msg = msg.merge(sess.dropna(subset=["session_id", "user_id"]).drop_duplicates("session_id"), on="session_id", how="left")

        if "role" in msg.columns:
            msg = msg[msg["role"].astype(str).str.lower().eq("user")]

        keep_cols = ["user_id", "message", "created_at"]
        if "session_id" in msg.columns:
            keep_cols.append("session_id")
        if "message_id" in msg.columns:
            keep_cols.append("message_id")
        if "sentiment_score" in msg.columns:
            keep_cols.append("sentiment_score")
        if "sentiment_label" in msg.columns:
            keep_cols.append("sentiment_label")
        msg = msg[keep_cols].copy()
        msg["source"] = "user_message"
        texts.append(msg)

    # Compatibility fallback: build sentiment from raw root CSVs when gnn_preprocessed is missing.
    if not texts and RAW_MESSAGES_PATH.exists() and RAW_SESSIONS_PATH.exists():
        msg = _read_csv_subset(
            RAW_MESSAGES_PATH,
            ["id", "session_id", "role", "message", "created_at", "sentiment_score", "sentiment_label"],
        )
        sess = _read_csv_subset(RAW_SESSIONS_PATH, ["id", "user_id"])

        sess = sess.rename(columns={"id": "session_id"})
        msg = msg.rename(columns={"id": "message_id"})

        msg["session_id"] = pd.to_numeric(msg.get("session_id"), errors="coerce")
        sess["session_id"] = pd.to_numeric(sess.get("session_id"), errors="coerce")
        sess["user_id"] = pd.to_numeric(sess.get("user_id"), errors="coerce")
        msg = msg.merge(sess.dropna(subset=["session_id", "user_id"]).drop_duplicates("session_id"), on="session_id", how="left")

        if "role" in msg.columns:
            msg = msg[msg["role"].astype(str).str.lower().eq("user")]

        keep_cols = ["user_id", "message", "created_at"]
        if "session_id" in msg.columns:
            keep_cols.append("session_id")
        if "message_id" in msg.columns:
            keep_cols.append("message_id")
        if "sentiment_score" in msg.columns:
            keep_cols.append("sentiment_score")
        if "sentiment_label" in msg.columns:
            keep_cols.append("sentiment_label")
        msg = msg[keep_cols].copy()
        msg["source"] = "user_message"
        texts.append(msg)

    if not texts:
        return pd.DataFrame(columns=["user_id", "message", "created_at", "source", "polarity", "subjectivity", "sentiment"])

    data = pd.concat(texts, ignore_index=True)
    data["user_id"] = pd.to_numeric(data["user_id"], errors="coerce")
    data = data.dropna(subset=["user_id"])
    data["user_id"] = data["user_id"].astype(int)
    data["message"] = data["message"].fillna("").astype(str).str.strip()
    data = data[data["message"].str.len() > 0]
    data["created_at"] = pd.to_datetime(data["created_at"], errors="coerce", utc=True)

    enable_hf_contextual = os.getenv("MAYA_ENABLE_CONTEXTUAL_HF", "0").strip().lower() in {"1", "true", "yes"}
    if enable_hf_contextual:
        contextual = contextual_hf_sentiment(data)
        if "sentiment_source" in contextual.columns and contextual["sentiment_source"].eq("huggingface_contextual").any():
            return contextual

    # Fallback path if HF models are unavailable.
    if "sentiment_score" in data.columns:
        data["polarity"] = pd.to_numeric(data["sentiment_score"], errors="coerce")
    else:
        data["polarity"] = np.nan

    missing_polarity = data["polarity"].isna()
    if missing_polarity.any():
        sent = data.loc[missing_polarity, "message"].apply(heuristic_sentiment_fallback)
        data.loc[missing_polarity, "polarity"] = sent.str[0].values
        data.loc[missing_polarity, "subjectivity"] = sent.str[1].values
    else:
        data["subjectivity"] = data["polarity"].astype(float).abs().clip(0.0, 1.0)

    data["polarity"] = pd.to_numeric(data["polarity"], errors="coerce").fillna(0.0)
    data = apply_negative_boost_from_text(data, text_col="message", polarity_col="polarity")
    data["subjectivity"] = pd.to_numeric(data["subjectivity"], errors="coerce").fillna(0.0)
    if "sentiment_label" in data.columns:
        lbl = data["sentiment_label"].fillna("").astype(str).str.lower().str.strip()
        data["sentiment"] = lbl.where(lbl.isin(["positive", "negative", "neutral"]), data["polarity"].apply(polarity_label))
    else:
        data["sentiment"] = data["polarity"].apply(polarity_label)
    data["sentiment_source"] = "heuristic_or_flink_fallback"
    return data


@st.cache_data(show_spinner=False)
def load_user_message_events() -> pd.DataFrame:
    msg_path = PREPROCESSED_DIR / "messages_nodes.csv"
    sess_path = PREPROCESSED_DIR / "sessions_nodes.csv"
    if msg_path.exists() and sess_path.exists():
        msg = _read_csv_subset(msg_path, ["message_id", "session_id", "created_at", "role", "message"])
        sess = _read_csv_subset(sess_path, ["session_id", "user_id", "has_transcription"])
    elif RAW_MESSAGES_PATH.exists() and RAW_SESSIONS_PATH.exists():
        msg = _read_csv_subset(RAW_MESSAGES_PATH, ["id", "session_id", "created_at", "role", "message"])
        sess = _read_csv_subset(RAW_SESSIONS_PATH, ["id", "user_id", "has_transcription"])
        msg = msg.rename(columns={"id": "message_id"})
        sess = sess.rename(columns={"id": "session_id"})
    else:
        return pd.DataFrame(columns=["message_id", "session_id", "user_id", "created_at", "role", "message", "has_transcription"])

    if "session_id" not in msg.columns or "session_id" not in sess.columns:
        return pd.DataFrame(columns=["message_id", "session_id", "user_id", "created_at", "role", "message", "has_transcription"])

    msg["session_id"] = pd.to_numeric(msg["session_id"], errors="coerce")
    sess["session_id"] = pd.to_numeric(sess["session_id"], errors="coerce")
    sess["user_id"] = pd.to_numeric(sess.get("user_id"), errors="coerce")
    if "message_id" in msg.columns:
        msg["message_id"] = pd.to_numeric(msg["message_id"], errors="coerce")
    else:
        msg["message_id"] = np.nan
    msg["created_at"] = pd.to_datetime(msg.get("created_at"), errors="coerce", utc=True)
    msg["role"] = msg.get("role", "").fillna("").astype(str).str.lower().str.strip()
    msg["message"] = msg.get("message", "").fillna("").astype(str)
    if "has_transcription" in sess.columns:
        sess["has_transcription"] = sess["has_transcription"].fillna(False).astype(bool)
    else:
        sess["has_transcription"] = False

    joined = msg.merge(
        sess[["session_id", "user_id", "has_transcription"]].dropna(subset=["session_id", "user_id"]).drop_duplicates("session_id"),
        on="session_id",
        how="left",
    )
    joined["user_id"] = pd.to_numeric(joined.get("user_id"), errors="coerce")
    joined = joined.dropna(subset=["session_id", "user_id", "created_at"]).copy()
    joined["session_id"] = joined["session_id"].astype(int)
    joined["user_id"] = joined["user_id"].astype(int)
    return joined[["message_id", "session_id", "user_id", "created_at", "role", "message", "has_transcription"]]


def build_latest_interaction_scores(user_sent: pd.DataFrame, top_n: int = 5) -> pd.DataFrame:
    if user_sent.empty:
        return pd.DataFrame(columns=["created_at", "emotion", "intent", "aspect"])

    df = user_sent.dropna(subset=["created_at"]).sort_values("created_at", ascending=False).head(top_n).copy()
    if df.empty:
        return pd.DataFrame(columns=["created_at", "emotion", "intent", "aspect"])

    def calc_intent(text: str) -> float:
        t = str(text)
        tokens = tokenize_message(t)
        has_action = any(tok in ACTION_VERBS for tok in tokens)
        is_question = ("?" in t) or bool(re.search(r"\b(what|why|how|can|could|would|when|where|who)\b", t.lower()))
        explicit_tasks = len(extract_task_candidates(t))
        score = 0.20 + 0.45 * float(has_action) + 0.20 * float(is_question) + 0.15 * min(explicit_tasks, 2) / 2.0
        return float(max(min(score, 1.0), 0.0))

    def calc_aspect(text: str) -> float:
        tokens = tokenize_message(text)
        if not tokens:
            return 0.0
        uniq_ratio = len(set(tokens)) / max(len(tokens), 1)
        density = min(len(tokens), 24) / 24.0
        task_density = min(len(extract_task_candidates(text)), 2) / 2.0
        score = 0.35 * uniq_ratio + 0.35 * density + 0.30 * task_density
        return float(max(min(score, 1.0), 0.0))

    out = df[["created_at", "message", "polarity", "subjectivity"]].copy()
    out["emotion"] = (
        0.70 * out["polarity"].astype(float).abs().clip(0, 1)
        + 0.30 * pd.to_numeric(out["subjectivity"], errors="coerce").fillna(0.0).clip(0, 1)
    ) * 100.0
    out["intent"] = out["message"].apply(calc_intent) * 100.0
    out["aspect"] = out["message"].apply(calc_aspect) * 100.0
    return out[["created_at", "emotion", "intent", "aspect"]].sort_values("created_at")


def build_response_sentiment_timeline(selected_user: int, user_sent: pd.DataFrame) -> pd.DataFrame:
    events = load_user_message_events()
    if events.empty:
        return pd.DataFrame(columns=["created_at", "response_time_sec", "polarity"])

    user_events = events[events["user_id"] == int(selected_user)].copy()
    if user_events.empty:
        return pd.DataFrame(columns=["created_at", "response_time_sec", "polarity"])

    user_events = user_events.sort_values(["session_id", "created_at"], kind="mergesort")
    user_events["next_role"] = user_events.groupby("session_id")["role"].shift(-1)
    user_events["next_time"] = user_events.groupby("session_id")["created_at"].shift(-1)
    paired = user_events[(user_events["role"] == "user") & (user_events["next_role"] == "assistant")].copy()
    if paired.empty:
        return pd.DataFrame(columns=["created_at", "response_time_sec", "polarity"])

    paired["response_time_sec"] = (paired["next_time"] - paired["created_at"]).dt.total_seconds()
    paired["response_time_sec"] = pd.to_numeric(paired["response_time_sec"], errors="coerce")
    paired = paired[(paired["response_time_sec"].notna()) & (paired["response_time_sec"] >= 0)].copy()
    if paired.empty:
        return pd.DataFrame(columns=["created_at", "response_time_sec", "polarity"])

    sent_cols = [c for c in ["message_id", "session_id", "created_at", "message", "polarity"] if c in user_sent.columns]
    sent_view = user_sent[sent_cols].copy() if sent_cols else pd.DataFrame()
    if not sent_view.empty:
        if "message_id" in paired.columns and "message_id" in sent_view.columns and sent_view["message_id"].notna().any():
            sent_view["message_id"] = pd.to_numeric(sent_view["message_id"], errors="coerce")
            paired = paired.merge(sent_view[["message_id", "polarity"]], on="message_id", how="left")
        else:
            sent_view["created_at_rounded"] = pd.to_datetime(sent_view["created_at"], errors="coerce", utc=True).dt.floor("s")
            paired["created_at_rounded"] = paired["created_at"].dt.floor("s")
            paired = paired.merge(sent_view[["created_at_rounded", "polarity"]], on="created_at_rounded", how="left")
    if "polarity" not in paired.columns:
        paired["polarity"] = 0.0

    paired["polarity"] = pd.to_numeric(paired["polarity"], errors="coerce").fillna(0.0).clip(-1.0, 1.0)
    return paired[["created_at", "response_time_sec", "polarity"]].sort_values("created_at")


def build_hri_metrics(selected_user: int, user_sent: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, bool]:
    events = load_user_message_events()
    user_events = events[(events["user_id"] == int(selected_user)) & (events["role"] == "user")].copy()

    if user_events.empty:
        modality = pd.DataFrame({"metric": ["Speech", "Touch"], "value": [0.0, 0.0]})
    else:
        speech_count = int(user_events["has_transcription"].fillna(False).sum())
        touch_count = int(len(user_events) - speech_count)
        if speech_count == 0 and touch_count > 0:
            voice_keywords = r"\b(call|voice|audio|speak|speaking|mic|microphone)\b"
            inferred_speech = int(user_events["message"].astype(str).str.contains(voice_keywords, case=False, regex=True).sum())
            speech_count = inferred_speech
            touch_count = max(len(user_events) - inferred_speech, 0)
        modality = pd.DataFrame({"metric": ["Speech", "Touch"], "value": [float(speech_count), float(touch_count)]})

    timeline = build_response_sentiment_timeline(selected_user, user_sent)
    proxy_used = True
    if timeline.empty:
        distance_score = 50.0
    else:
        avg_rt = float(timeline["response_time_sec"].median())
        distance_score = float(max(min(100.0 * _sigmoid((45.0 - avg_rt) / 18.0), 100.0), 0.0))

    if user_sent.empty:
        posture_score = 50.0
    else:
        volatility = float(user_sent.sort_values("created_at")["polarity"].diff().abs().mean() or 0.0)
        posture_score = float(max(min((1.0 - min(volatility, 1.0)) * 100.0, 100.0), 0.0))

    physical = pd.DataFrame(
        {"metric": ["Distance", "Posture"], "value": [distance_score, posture_score]}
    )
    return modality, physical, proxy_used


def tokenize_message(text: str) -> list[str]:
    tokens = re.findall(r"[a-zA-Z][a-zA-Z']{1,}", str(text).lower())
    return [t for t in tokens if t not in STOPWORDS and len(t) > 2]


def normalize_task_phrase(text: str) -> str:
    tokens = [
        t for t in re.findall(r"[a-zA-Z][a-zA-Z']{1,}", str(text).lower())
        if t not in STOPWORDS and t not in FILLER_WORDS
    ]
    if not tokens:
        return ""

    if tokens[0] not in ACTION_VERBS and len(tokens) > 1:
        for i, t in enumerate(tokens):
            if t in ACTION_VERBS:
                tokens = tokens[i:]
                break

    tokens = tokens[:6]
    return " ".join(tokens).strip()


def extract_task_candidates(message: str) -> list[str]:
    text = str(message).lower()
    candidates: list[str] = []

    for pattern in TASK_PATTERNS:
        for m in pattern.finditer(text):
            phrase = normalize_task_phrase(m.group(1))
            if phrase:
                candidates.append(phrase)

    toks = [t for t in re.findall(r"[a-zA-Z][a-zA-Z']{1,}", text)]
    if toks:
        if toks[0] in ACTION_VERBS:
            phrase = normalize_task_phrase(" ".join(toks[:6]))
            if phrase:
                candidates.append(phrase)

        for i in range(len(toks) - 1):
            if toks[i] in ACTION_VERBS and toks[i + 1] not in STOPWORDS and toks[i + 1] not in FILLER_WORDS:
                tail = toks[i : min(i + 5, len(toks))]
                phrase = normalize_task_phrase(" ".join(tail))
                if phrase:
                    candidates.append(phrase)

    # De-duplicate while preserving order.
    unique = []
    seen = set()
    for c in candidates:
        if c not in seen and len(c.split()) >= 2:
            seen.add(c)
            unique.append(c)
    return unique


def infer_canonical_intents(message: str) -> list[str]:
    text = str(message).lower().strip()
    if not text:
        return []
    intents: list[str] = []
    for label, patterns in CANONICAL_INTENT_PATTERNS.items():
        if any(p.search(text) for p in patterns):
            intents.append(label)
    return intents


def build_task_importance(sentiment_df: pd.DataFrame, user_id: int | None = None, top_k: int = 15) -> pd.DataFrame:
    if sentiment_df.empty:
        return pd.DataFrame(columns=["task", "mentions", "avg_polarity", "importance", "sample_request"])

    df = sentiment_df.copy()
    if user_id is not None:
        df = df[df["user_id"] == user_id].copy()
    if df.empty:
        return pd.DataFrame(columns=["task", "mentions", "avg_polarity", "importance", "sample_request"])

    task_counts: Counter[str] = Counter()
    task_polarity: dict[str, list[float]] = {}
    task_examples: dict[str, str] = {}

    for _, row in df.iterrows():
        message = str(row.get("message", ""))
        candidates = infer_canonical_intents(message)
        if not candidates:
            candidates = extract_task_candidates(message)
        if not candidates:
            continue
        pol = float(row.get("polarity", 0.0))

        for task in candidates:
            task_counts[task] += 1
            task_polarity.setdefault(task, []).append(pol)
            if task not in task_examples:
                task_examples[task] = message

    if not task_counts:
        return pd.DataFrame(columns=["task", "mentions", "avg_polarity", "importance", "sample_request"])

    rows = []
    for task, count in task_counts.most_common(top_k * 4):
        pol_vals = task_polarity.get(task, [0.0])
        avg_pol = float(np.mean(pol_vals))
        mean_abs_polarity = float(np.mean(np.abs(pol_vals)))
        # Rank mainly by repeated asks, then by emotional intensity.
        importance = float(count * (1.0 + mean_abs_polarity))
        rows.append(
            {
                "task": task,
                "mentions": int(count),
                "avg_polarity": avg_pol,
                "importance": importance,
                "sample_request": task_examples.get(task, ""),
            }
        )

    out = pd.DataFrame(rows).sort_values(["importance", "mentions"], ascending=False).head(top_k)
    return out


def build_representative_statements(sentiment_df: pd.DataFrame, user_id: int | None = None, top_k: int = 10) -> pd.DataFrame:
    if sentiment_df.empty:
        return pd.DataFrame(columns=["created_at", "message", "polarity", "sentiment", "context_score"])

    df = sentiment_df.copy()
    if user_id is not None:
        df = df[df["user_id"] == user_id].copy()
    if df.empty:
        return pd.DataFrame(columns=["created_at", "message", "polarity", "sentiment", "context_score"])

    top_tasks = set(build_task_importance(df, None, top_k=20)["task"].tolist())

    def statement_score(row: pd.Series) -> float:
        msg = str(row.get("message", ""))
        tokens = tokenize_message(msg)
        if not tokens:
            return 0.0
        token_set = set(tokens)
        task_hits = 0
        token_text = " ".join(tokens)
        for task in top_tasks:
            p_parts = task.split()
            if len(p_parts) == 1:
                if p_parts[0] in token_set:
                    task_hits += 1
            else:
                if task in token_text:
                    task_hits += 1
        len_score = min(len(tokens) / 18.0, 1.0)
        sentiment_strength = min(abs(float(row.get("polarity", 0.0))), 1.0)
        return float(0.55 * (task_hits / max(len(top_tasks), 1)) + 0.30 * sentiment_strength + 0.15 * len_score)

    scored = df.copy()
    scored["context_score"] = scored.apply(statement_score, axis=1)
    scored = scored.sort_values(["context_score", "created_at"], ascending=[False, False]).head(top_k)
    return scored[["created_at", "message", "polarity", "sentiment", "context_score"]]


def _map_request_to_feature_focus(text: str) -> str:
    low = str(text).lower().strip()
    if not low:
        return "Other"
    for feature_name, patterns in FEATURE_FOCUS_PATTERNS.items():
        if any(p.search(low) for p in patterns):
            return feature_name
    return "Other"


def build_feature_focus_summary(sentiment_df: pd.DataFrame, top_k: int = 10) -> pd.DataFrame:
    if sentiment_df.empty:
        return pd.DataFrame(columns=["feature_focus", "mentions", "share", "avg_polarity", "sample_requests"])

    rows: list[dict[str, object]] = []
    for _, row in sentiment_df.iterrows():
        msg = str(row.get("message", ""))
        pol = float(row.get("polarity", 0.0))
        candidates = extract_task_candidates(msg)
        if not candidates:
            continue
        for c in candidates:
            rows.append(
                {
                    "feature_focus": _map_request_to_feature_focus(c),
                    "task": c,
                    "message": msg,
                    "polarity": pol,
                }
            )

    if not rows:
        return pd.DataFrame(columns=["feature_focus", "mentions", "share", "avg_polarity", "sample_requests"])

    req = pd.DataFrame(rows)
    total_mentions = float(len(req))
    grouped = req.groupby("feature_focus", as_index=False).agg(
        mentions=("task", "size"),
        avg_polarity=("polarity", "mean"),
    )
    grouped["share"] = grouped["mentions"] / max(total_mentions, 1.0)

    top_examples = (
        req.groupby("feature_focus")["task"]
        .apply(lambda s: ", ".join(pd.Series(s).value_counts().head(3).index.tolist()))
        .reset_index(name="sample_requests")
    )
    out = grouped.merge(top_examples, on="feature_focus", how="left")
    out = out.sort_values(["mentions", "share"], ascending=False).head(top_k)
    return out[["feature_focus", "mentions", "share", "avg_polarity", "sample_requests"]]


def _scale_0_1(s: pd.Series) -> pd.Series:
    x = pd.to_numeric(s, errors="coerce").fillna(0.0).astype(float)
    if x.empty:
        return x
    vmin = float(x.min())
    vmax = float(x.max())
    if np.isclose(vmax - vmin, 0.0):
        return pd.Series(np.zeros(len(x), dtype=float), index=x.index)
    return (x - vmin) / (vmax - vmin)


def build_rag_roadmap_signals(sentiment_df: pd.DataFrame, recent_days: int = 30, top_k: int = 12) -> pd.DataFrame:
    cols = [
        "intent",
        "mentions",
        "share",
        "avg_polarity",
        "neg_ratio",
        "recent_mentions",
        "previous_mentions",
        "trend_pct",
        "opportunity_score",
        "sample_requests",
    ]
    if sentiment_df.empty:
        return pd.DataFrame(columns=cols)

    rows: list[dict[str, object]] = []
    for _, row in sentiment_df.iterrows():
        msg = str(row.get("message", "")).strip()
        if not msg:
            continue
        intents = infer_canonical_intents(msg)
        if not intents:
            continue
        for intent in sorted(set(intents)):
            rows.append(
                {
                    "intent": intent,
                    "message": msg,
                    "created_at": row.get("created_at"),
                    "polarity": float(row.get("polarity", 0.0)),
                }
            )

    if not rows:
        return pd.DataFrame(columns=cols)

    req = pd.DataFrame(rows)
    req["created_at"] = pd.to_datetime(req["created_at"], errors="coerce", utc=True)
    req["is_negative"] = req["polarity"] < 0

    total_mentions = float(len(req))
    out = req.groupby("intent", as_index=False).agg(
        mentions=("intent", "size"),
        avg_polarity=("polarity", "mean"),
        neg_ratio=("is_negative", "mean"),
    )
    out["share"] = out["mentions"] / max(total_mentions, 1.0)

    if req["created_at"].notna().any():
        anchor = req["created_at"].max()
        recent_cut = anchor - pd.Timedelta(days=recent_days)
        prev_cut = recent_cut - pd.Timedelta(days=recent_days)
        recent_counts = (
            req[req["created_at"] >= recent_cut]
            .groupby("intent")
            .size()
            .rename("recent_mentions")
            .reset_index()
        )
        prev_counts = (
            req[(req["created_at"] < recent_cut) & (req["created_at"] >= prev_cut)]
            .groupby("intent")
            .size()
            .rename("previous_mentions")
            .reset_index()
        )
        out = out.merge(recent_counts, on="intent", how="left").merge(prev_counts, on="intent", how="left")
    else:
        out["recent_mentions"] = 0
        out["previous_mentions"] = 0

    out["recent_mentions"] = pd.to_numeric(out.get("recent_mentions"), errors="coerce").fillna(0).astype(int)
    out["previous_mentions"] = pd.to_numeric(out.get("previous_mentions"), errors="coerce").fillna(0).astype(int)
    out["trend_pct"] = (
        (out["recent_mentions"] - out["previous_mentions"])
        / out["previous_mentions"].replace(0, np.nan)
    ).replace([np.inf, -np.inf], np.nan).fillna(1.0)

    demand_norm = _scale_0_1(out["mentions"])
    neg_strength = (-out["avg_polarity"]).clip(lower=0.0)
    diss_norm = 0.6 * _scale_0_1(out["neg_ratio"]) + 0.4 * _scale_0_1(neg_strength)
    trend_norm = _scale_0_1(out["trend_pct"])
    out["opportunity_score"] = (100.0 * (0.50 * demand_norm + 0.35 * diss_norm + 0.15 * trend_norm)).round(2)

    samples = (
        req.groupby("intent")["message"]
        .apply(lambda s: " | ".join(pd.Series(s).dropna().astype(str).head(3).tolist()))
        .reset_index(name="sample_requests")
    )
    out = out.merge(samples, on="intent", how="left")
    out = out.sort_values(["opportunity_score", "mentions"], ascending=False).head(top_k)
    return out[cols]


@st.cache_data(show_spinner=False)
def build_user_snapshot(user_id: int, refresh_nonce: int) -> Tuple[pd.DataFrame, pd.DataFrame]:
    sentiment_df = load_sentiment_table()
    user_sent = sentiment_df[sentiment_df["user_id"] == user_id].copy()
    task_imp = build_task_importance(sentiment_df, user_id=user_id, top_k=15)
    return user_sent, task_imp


@st.cache_data(show_spinner=False)
def load_xgb_shap_importance() -> pd.DataFrame:
    if XGB_SHAP_IMPORTANCE_PATH.exists():
        imp = pd.read_csv(XGB_SHAP_IMPORTANCE_PATH)
        if {"feature", "mean_abs_shap"}.issubset(set(imp.columns)):
            imp["mean_abs_shap"] = pd.to_numeric(imp["mean_abs_shap"], errors="coerce").fillna(0.0)
            return imp.sort_values("mean_abs_shap", ascending=False)

    imp_r = load_df_from_redis(
        "xgb_embedding_feature_importance",
        expected_cols=["feature", "mean_abs_shap", "feature_label"],
    )
    if not imp_r.empty:
        imp_r["mean_abs_shap"] = pd.to_numeric(imp_r["mean_abs_shap"], errors="coerce").fillna(0.0)
        return imp_r.sort_values("mean_abs_shap", ascending=False)
    return pd.DataFrame(columns=["feature", "mean_abs_shap"])


@st.cache_data(show_spinner=False)
def load_xgb_target_report() -> pd.DataFrame:
    expected = [
        "target_source",
        "human_label_column",
        "human_label_rows",
        "pseudo_label_rows",
        "joined_users",
        "train_rows",
        "test_rows",
        "accuracy",
        "auc",
        "warning",
    ]
    p = XGB_ARTIFACT_DIR / "xgb_target_report.csv"
    if p.exists():
        rep = pd.read_csv(p)
    else:
        rep_r = load_df_from_redis("xgb_target_report", expected_cols=expected)
        if rep_r.empty:
            return pd.DataFrame(columns=expected)
        rep = rep_r.copy()

    for c in ["human_label_rows", "pseudo_label_rows", "joined_users", "train_rows", "test_rows"]:
        if c in rep.columns:
            rep[c] = pd.to_numeric(rep[c], errors="coerce").fillna(0).astype(int)
    for c in ["accuracy", "auc"]:
        if c in rep.columns:
            rep[c] = pd.to_numeric(rep[c], errors="coerce")
    return rep


@st.cache_data(show_spinner=False)
def load_xgb_user_predictions() -> pd.DataFrame:
    expected = ["user_id", "target", "pred_label", "pred_prob_positive", "predicted_class", "confidence"]
    if XGB_PREDICTIONS_PATH.exists():
        df = pd.read_csv(XGB_PREDICTIONS_PATH)
    else:
        df_r = load_df_from_redis("xgb_user_predictions", expected_cols=expected)
        if df_r.empty:
            return pd.DataFrame(columns=expected)
        df = df_r.copy()

    if "user_id" in df.columns:
        df["user_id"] = pd.to_numeric(df["user_id"], errors="coerce")
        df = df.dropna(subset=["user_id"]).copy()
        df["user_id"] = df["user_id"].astype(int)
    for c in ["target", "pred_label"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0).astype(int)
    for c in ["pred_prob_positive", "confidence"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0.0)
    if "predicted_class" not in df.columns and "pred_label" in df.columns:
        df["predicted_class"] = np.where(df["pred_label"] == 1, "positive", "negative")
    return df.sort_values("pred_prob_positive", ascending=False)


@st.cache_data(show_spinner=False)
def load_embedding_dimension_labels() -> pd.DataFrame:
    if EMBEDDING_LABELS_PATH.exists():
        out = pd.read_csv(EMBEDDING_LABELS_PATH)
    else:
        lbl_r = load_df_from_redis(
            "embedding_dimension_labels",
            expected_cols=["feature", "label", "anchor_feature", "anchor_feature_label", "abs_correlation"],
        )
        out = lbl_r.copy() if not lbl_r.empty else pd.DataFrame()

    if out.empty:
        # Fallback: derive semantic labels live from embeddings + reconstructed user features.
        out = derive_embedding_dimension_labels_live()
        if out.empty:
            return pd.DataFrame(columns=["feature", "label", "anchor_feature", "anchor_feature_label", "abs_correlation", "top_signals"])

    out["feature"] = out.get("feature", "").fillna("").astype(str)
    out["label"] = out.get("label", "").fillna("").astype(str)
    out["anchor_feature"] = out.get("anchor_feature", "").fillna("").astype(str)
    out["anchor_feature_label"] = out.get("anchor_feature_label", "").fillna("").astype(str)
    out["abs_correlation"] = pd.to_numeric(out.get("abs_correlation"), errors="coerce").fillna(0.0)
    if "top_signals" not in out.columns:
        out["top_signals"] = ""
    return out.drop_duplicates("feature", keep="last")


@st.cache_data(show_spinner=False)
def derive_embedding_dimension_labels_live() -> pd.DataFrame:
    try:
        from pipelines.training.train_user_behavior_gnn import build_user_table, build_feature_matrix
    except Exception:
        return pd.DataFrame(
            columns=["feature", "label", "anchor_feature", "anchor_feature_label", "abs_correlation", "top_signals"]
        )

    emb = load_embeddings_df()
    if emb.empty or "user_id" not in emb.columns:
        return pd.DataFrame(
            columns=["feature", "label", "anchor_feature", "anchor_feature_label", "abs_correlation", "top_signals"]
        )
    emb_cols = [c for c in emb.columns if str(c).startswith("emb_")]
    if not emb_cols:
        return pd.DataFrame(
            columns=["feature", "label", "anchor_feature", "anchor_feature_label", "abs_correlation", "top_signals"]
        )

    users = load_df_from_redis("users_nodes")
    sessions = load_df_from_redis("sessions_nodes")
    messages = load_df_from_redis("messages_nodes")
    feedback = load_df_from_redis("feedback_nodes")

    if users.empty:
        p = PREPROCESSED_DIR / "users_nodes.csv"
        if p.exists():
            users = pd.read_csv(p)
    if sessions.empty:
        p = PREPROCESSED_DIR / "sessions_nodes.csv"
        if p.exists():
            sessions = pd.read_csv(p)
    if messages.empty:
        p = PREPROCESSED_DIR / "messages_nodes.csv"
        if p.exists():
            messages = pd.read_csv(p)
    if feedback.empty:
        p = PREPROCESSED_DIR / "feedback_nodes.csv"
        if p.exists():
            feedback = pd.read_csv(p)

    if users.empty or sessions.empty or messages.empty or feedback.empty:
        return pd.DataFrame(
            columns=["feature", "label", "anchor_feature", "anchor_feature_label", "abs_correlation", "top_signals"]
        )

    try:
        user_df = build_user_table(users, sessions, messages, feedback)
        feature_df, feature_names = build_feature_matrix(user_df)
    except Exception:
        return pd.DataFrame(
            columns=["feature", "label", "anchor_feature", "anchor_feature_label", "abs_correlation", "top_signals"]
        )

    joined = emb.merge(user_df[["user_id"]], on="user_id", how="inner")
    if joined.empty:
        return pd.DataFrame(
            columns=["feature", "label", "anchor_feature", "anchor_feature_label", "abs_correlation", "top_signals"]
        )

    feat_aligned = feature_df.copy()
    feat_aligned["user_id"] = user_df["user_id"].astype(int).values
    joined = joined.merge(feat_aligned, on="user_id", how="inner")
    if joined.empty:
        return pd.DataFrame(
            columns=["feature", "label", "anchor_feature", "anchor_feature_label", "abs_correlation", "top_signals"]
        )

    E = joined[emb_cols].apply(pd.to_numeric, errors="coerce").fillna(0.0).to_numpy(dtype=np.float64)
    F = joined[feature_names].apply(pd.to_numeric, errors="coerce").fillna(0.0).to_numpy(dtype=np.float64)
    n = E.shape[0]
    if n < 2:
        return pd.DataFrame(
            columns=["feature", "label", "anchor_feature", "anchor_feature_label", "abs_correlation", "top_signals"]
        )

    e_std = E.std(axis=0, keepdims=True)
    f_std = F.std(axis=0, keepdims=True)
    e_std[e_std == 0] = 1.0
    f_std[f_std == 0] = 1.0
    E_z = (E - E.mean(axis=0, keepdims=True)) / e_std
    F_z = (F - F.mean(axis=0, keepdims=True)) / f_std
    corr = np.abs((E_z.T @ F_z) / float(max(n - 1, 1)))

    rows: list[dict[str, object]] = []
    for emb_i, emb_name in enumerate(emb_cols):
        c_row = corr[emb_i]
        top_idx = np.argsort(-c_row)[:3]
        anchor_idx = int(top_idx[0])
        anchor_feat = str(feature_names[anchor_idx])
        anchor_label = humanize_feature_name(anchor_feat)
        top_signals = ", ".join(humanize_feature_name(str(feature_names[j])) for j in top_idx)
        rows.append(
            {
                "feature": emb_name,
                "label": f"{prettify_embedding_feature_name(emb_name)} - {anchor_label}",
                "anchor_feature": anchor_feat,
                "anchor_feature_label": anchor_label,
                "abs_correlation": float(c_row[anchor_idx]),
                "top_signals": top_signals,
            }
        )
    return pd.DataFrame(rows).sort_values("feature")


@st.cache_data(show_spinner=False)
def embedding_shape() -> tuple[int, int]:
    if not USER_EMBEDDINGS_PATH.exists():
        return 0, 0
    emb = pd.read_csv(USER_EMBEDDINGS_PATH)
    cols = [c for c in emb.columns if str(c).startswith("emb_")]
    return int(len(emb)), int(len(cols))


@st.cache_data(show_spinner=False)
def load_embeddings_df() -> pd.DataFrame:
    emb_r = load_df_from_redis("user_embeddings")
    if not emb_r.empty:
        emb = emb_r.copy()
        if "user_id" in emb.columns:
            emb["user_id"] = pd.to_numeric(emb["user_id"], errors="coerce")
            emb = emb.dropna(subset=["user_id"]).copy()
            emb["user_id"] = emb["user_id"].astype(int)
        return emb

    if not USER_EMBEDDINGS_PATH.exists():
        return pd.DataFrame(columns=["user_id"])
    emb = pd.read_csv(USER_EMBEDDINGS_PATH)
    if "user_id" in emb.columns:
        emb["user_id"] = pd.to_numeric(emb["user_id"], errors="coerce")
        emb = emb.dropna(subset=["user_id"]).copy()
        emb["user_id"] = emb["user_id"].astype(int)
    return emb


@st.cache_data(show_spinner=False)
def build_tsne_persona(persona_table: pd.DataFrame) -> pd.DataFrame:
    emb = load_embeddings_df()
    if emb.empty or "user_id" not in emb.columns or persona_table.empty:
        return pd.DataFrame(columns=["user_id", "persona_label", "tsne_x", "tsne_y"])

    emb_cols = [c for c in emb.columns if str(c).startswith("emb_")]
    if not emb_cols:
        return pd.DataFrame(columns=["user_id", "persona_label", "tsne_x", "tsne_y"])

    joined = emb.merge(persona_table[["user_id", "persona_label"]], on="user_id", how="inner")
    if joined.empty:
        return pd.DataFrame(columns=["user_id", "persona_label", "tsne_x", "tsne_y"])

    max_points = int(os.getenv("MAYA_TSNE_MAX_POINTS", "500"))
    if len(joined) > max_points:
        joined = joined.sample(n=max_points, random_state=42).copy()

    X = joined[emb_cols].apply(pd.to_numeric, errors="coerce").fillna(0.0).to_numpy(dtype=np.float32)
    n = X.shape[0]
    if n < 3:
        return pd.DataFrame(columns=["user_id", "persona_label", "tsne_x", "tsne_y"])

    perp = max(2, min(30, n - 1))
    tsne = TSNE(n_components=2, random_state=42, perplexity=perp, init="pca", learning_rate="auto")
    xy = tsne.fit_transform(X)
    out = joined[["user_id", "persona_label"]].copy()
    out["tsne_x"] = xy[:, 0]
    out["tsne_y"] = xy[:, 1]
    return out


def summarize_persona_reasons(persona_table: pd.DataFrame) -> pd.DataFrame:
    if persona_table.empty:
        return pd.DataFrame(columns=["persona_label", "top_reasons_summary"])

    rows = []
    reason_cols = [c for c in ["top_reason_1", "top_reason_2", "top_reason_3"] if c in persona_table.columns]
    for persona, grp in persona_table.groupby("persona_label"):
        vals = []
        for c in reason_cols:
            vals.extend(grp[c].dropna().astype(str).tolist())
        if vals:
            vc = pd.Series(vals).value_counts().head(3).index.tolist()
            summary = ", ".join(vc)
        else:
            summary = "n/a"
        rows.append({"persona_label": persona, "top_reasons_summary": summary})
    return pd.DataFrame(rows).sort_values("persona_label")


@st.cache_data(show_spinner=False)
def load_persona_outputs() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    table = load_df_from_redis(
        "user_persona_table",
        expected_cols=["user_id", "persona_label", "top_reason_1", "top_reason_2", "top_reason_3"],
    )
    profiles = load_df_from_redis(
        "persona_profiles",
        expected_cols=["persona_id", "users", "avg_sentiment", "account_age_days", "msg_count", "persona_label"],
    )
    importance = load_df_from_redis("persona_feature_importance", expected_cols=["feature", "importance"])

    if table.empty and PERSONA_TABLE_PATH.exists():
        table = pd.read_csv(PERSONA_TABLE_PATH)
    if profiles.empty and PERSONA_PROFILE_PATH.exists():
        profiles = pd.read_csv(PERSONA_PROFILE_PATH)
    if importance.empty and PERSONA_IMPORTANCE_PATH.exists():
        importance = pd.read_csv(PERSONA_IMPORTANCE_PATH)

    if "user_id" in table.columns:
        table["user_id"] = pd.to_numeric(table["user_id"], errors="coerce").fillna(-1).astype(int)
    if "users" in profiles.columns:
        profiles["users"] = pd.to_numeric(profiles["users"], errors="coerce").fillna(0).astype(int)
    if "importance" in importance.columns:
        importance["importance"] = pd.to_numeric(importance["importance"], errors="coerce").fillna(0.0)

    return table, profiles, importance


@st.cache_data(show_spinner=False)
def load_persona_user_shap() -> pd.DataFrame:
    df_r = load_df_from_redis(
        "persona_user_feature_contributions",
        expected_cols=["user_id", "persona_id", "persona_label", "feature", "shap_value", "abs_shap"],
    )
    if not df_r.empty:
        df = df_r.copy()
        for c in ["user_id", "persona_id"]:
            if c in df.columns:
                df[c] = pd.to_numeric(df[c], errors="coerce").fillna(-1).astype(int)
        for c in ["shap_value", "abs_shap"]:
            if c in df.columns:
                df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0.0)
        return df

    if not PERSONA_USER_SHAP_PATH.exists():
        return pd.DataFrame(columns=["user_id", "persona_id", "persona_label", "feature", "shap_value", "abs_shap"])

    df = pd.read_csv(PERSONA_USER_SHAP_PATH)
    for c in ["user_id", "persona_id"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce").fillna(-1).astype(int)
    for c in ["shap_value", "abs_shap"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0.0)
    return df


@st.cache_data(show_spinner=False)
def load_user_dissatisfaction_flags() -> pd.DataFrame:
    cols = ["user_id", "avg_sentiment", "neg_ratio", "msg_count", "dissatisfaction_score", "dissatisfaction_flag", "dissatisfaction_reason"]
    if SENTIMENT_SCORES_PATH.exists():
        s = pd.read_csv(SENTIMENT_SCORES_PATH)
    else:
        msg_nodes = PREPROCESSED_DIR / "messages_nodes.csv"
        if not msg_nodes.exists():
            return pd.DataFrame(columns=cols)
        s = pd.read_csv(msg_nodes)
    if "role" in s.columns:
        role = s["role"].astype(str).str.lower().str.strip()
        if role.eq("user").any():
            s = s[role.eq("user")].copy()

    s = repair_flat_sentiment_scores(s, text_col="message", score_col="sentiment_score", label_col="sentiment_label")

    if "user_id" not in s.columns:
        if "session_id" in s.columns and SESSIONS_SOURCE_PATH.exists():
            sess = pd.read_csv(SESSIONS_SOURCE_PATH, usecols=["id", "user_id"])
            sess["id"] = pd.to_numeric(sess["id"], errors="coerce")
            sess["user_id"] = pd.to_numeric(sess["user_id"], errors="coerce")
            s["session_id"] = pd.to_numeric(s["session_id"], errors="coerce")
            s = s.merge(sess.rename(columns={"id": "session_id"}), on="session_id", how="left")
        else:
            return pd.DataFrame(columns=cols)

    s["user_id"] = pd.to_numeric(s.get("user_id"), errors="coerce")
    s = s.dropna(subset=["user_id"]).copy()
    s["user_id"] = s["user_id"].astype(int)

    lbl_col = "sentiment_label" if "sentiment_label" in s.columns else "sentiment"
    if lbl_col not in s.columns:
        s[lbl_col] = "neutral"
    s[lbl_col] = s[lbl_col].astype(str).str.lower().str.strip()

    s["sentiment_score"] = pd.to_numeric(s.get("sentiment_score"), errors="coerce").fillna(0.0)

    agg = s.groupby("user_id", as_index=False).agg(
        avg_sentiment=("sentiment_score", "mean"),
        msg_count=("sentiment_score", "size"),
        neg_ratio=(lbl_col, lambda x: float((x == "negative").mean())),
    )

    neg_strength = (-agg["avg_sentiment"]).clip(lower=0.0)
    if neg_strength.max() > 0:
        neg_strength = neg_strength / neg_strength.max()
    neg_ratio_scaled = agg["neg_ratio"]
    if neg_ratio_scaled.max() > 0:
        neg_ratio_scaled = neg_ratio_scaled / neg_ratio_scaled.max()

    agg["dissatisfaction_score"] = 0.55 * neg_ratio_scaled + 0.45 * neg_strength

    q80 = float(agg["dissatisfaction_score"].quantile(0.80)) if not agg.empty else 0.0
    q60 = float(agg["dissatisfaction_score"].quantile(0.60)) if not agg.empty else 0.0

    def bucket(v: float) -> str:
        if v >= q80:
            return "High"
        if v >= q60:
            return "Medium"
        return "Low"

    agg["dissatisfaction_flag"] = agg["dissatisfaction_score"].apply(bucket)

    def reason(row: pd.Series) -> str:
        if row["neg_ratio"] >= 0.10:
            return "higher share of negative messages"
        if row["avg_sentiment"] < -0.01:
            return "overall negative sentiment trend"
        if row["msg_count"] >= float(agg["msg_count"].quantile(0.75)):
            return "high-volume interactions with mixed sentiment"
        return "mildly negative relative to peers"

    agg["dissatisfaction_reason"] = agg.apply(reason, axis=1)
    return agg[cols]


@st.cache_data(show_spinner=False)
def load_whatsapp_sentiment_messages() -> pd.DataFrame:
    cols = ["user_id", "message", "created_at", "sentiment_score", "sentiment_label", "role"]
    if SENTIMENT_SCORES_PATH.exists():
        s = pd.read_csv(SENTIMENT_SCORES_PATH)
    else:
        msg_nodes = PREPROCESSED_DIR / "messages_nodes.csv"
        if not msg_nodes.exists():
            return pd.DataFrame(columns=cols)
        s = pd.read_csv(msg_nodes)
    if "role" in s.columns:
        role = s["role"].astype(str).str.lower().str.strip()
        if role.eq("user").any():
            s = s[role.eq("user")].copy()

    s = repair_flat_sentiment_scores(s, text_col="message", score_col="sentiment_score", label_col="sentiment_label")

    if "user_id" not in s.columns:
        if "session_id" in s.columns and SESSIONS_SOURCE_PATH.exists():
            sess = pd.read_csv(SESSIONS_SOURCE_PATH, usecols=["id", "user_id"])
            sess["id"] = pd.to_numeric(sess["id"], errors="coerce")
            sess["user_id"] = pd.to_numeric(sess["user_id"], errors="coerce")
            s["session_id"] = pd.to_numeric(s["session_id"], errors="coerce")
            s = s.merge(sess.rename(columns={"id": "session_id"}), on="session_id", how="left")
        else:
            return pd.DataFrame(columns=cols)

    s["user_id"] = pd.to_numeric(s.get("user_id"), errors="coerce")
    s = s.dropna(subset=["user_id"]).copy()
    s["user_id"] = s["user_id"].astype(int)
    s["message"] = s.get("message", "").fillna("").astype(str)
    s["created_at"] = pd.to_datetime(s.get("created_at"), errors="coerce", utc=True)
    s["sentiment_score"] = pd.to_numeric(s.get("sentiment_score"), errors="coerce").fillna(0.0)

    if "sentiment_label" not in s.columns:
        s["sentiment_label"] = s["sentiment_score"].apply(polarity_label)
    else:
        s["sentiment_label"] = s["sentiment_label"].fillna("").astype(str).str.lower().str.strip()
        s["sentiment_label"] = s["sentiment_label"].replace({"": "neutral"})

    if "role" not in s.columns:
        s["role"] = "user"
    return s[cols]


@st.cache_data(show_spinner=False)
def load_gru_mood_swing_summary() -> pd.DataFrame:
    expected = [
        "user_id",
        "messages",
        "actual_volatility",
        "predicted_volatility",
        "prediction_mae",
        "mood_swing_index",
        "risk_flag",
        "trend",
        "recommendation",
    ]
    if GRU_MOOD_SWING_SUMMARY_PATH.exists():
        df = pd.read_csv(GRU_MOOD_SWING_SUMMARY_PATH)
    else:
        df = load_df_from_redis("gru_mood_swing_summary", expected_cols=expected)
        if df.empty:
            return pd.DataFrame(columns=expected)

    for c in ["user_id", "messages"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0).astype(int)
    for c in ["actual_volatility", "predicted_volatility", "prediction_mae", "mood_swing_index"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0.0)
    return df


@st.cache_data(show_spinner=False)
def load_gru_mood_training_report() -> pd.DataFrame:
    expected = [
        "total_messages",
        "eligible_users",
        "sequence_length",
        "hidden_size",
        "epochs",
        "batch_size",
        "train_samples",
        "val_samples",
        "train_loss",
        "val_mse",
    ]
    if GRU_MOOD_TRAINING_REPORT_PATH.exists():
        df = pd.read_csv(GRU_MOOD_TRAINING_REPORT_PATH)
    else:
        df = load_df_from_redis("gru_mood_training_report", expected_cols=expected)
        if df.empty:
            return pd.DataFrame(columns=expected)

    return df


def run_gru_mood_training_action() -> tuple[bool, str]:
    cmd = [sys.executable, "-m", "pipelines.training.train_whatsapp_gru_mood_swings"]
    try:
        proc = subprocess.run(
            cmd,
            cwd=str(BASE_DIR),
            capture_output=True,
            text=True,
            check=False,
        )
    except Exception as exc:
        return False, f"Failed to launch GRU training command: {exc}"

    stdout = (proc.stdout or "").strip()
    stderr = (proc.stderr or "").strip()
    combined = "\n".join(part for part in [stdout, stderr] if part)
    if not combined:
        combined = "No output captured from training process."
    return proc.returncode == 0, combined


def pipeline_steps_for_ui() -> list[dict[str, str]]:
    try:
        from run_pipeline import build_steps  # lazy import to avoid hard dependency during module import

        steps = build_steps(include_redis_publish=False)
        return [
            {
                "step_id": s.id,
                "description": s.description,
                "command": " ".join(s.cmd),
            }
            for s in steps
        ]
    except Exception:
        # Fallback list if orchestrator import is unavailable for any reason.
        py = sys.executable
        return [
            {"step_id": "feature_engineering", "description": "Build user-level engineered feature matrix", "command": f"{py} -m pipelines.preprocessing.feature_engineering"},
            {"step_id": "train_graphsage_user_embeddings", "description": "Train GraphSAGE embeddings", "command": f"{py} -m pipelines.training.train_graphsage_user_embeddings"},
            {"step_id": "build_gnn_nodes_from_flink", "description": "Build GNN node tables from Flink outputs", "command": f"{py} -m pipelines.preprocessing.build_gnn_nodes_from_flink"},
            {"step_id": "train_user_behavior_gnn", "description": "Train user behavior GNN", "command": f"{py} -m pipelines.training.train_user_behavior_gnn"},
            {"step_id": "train_xgb_shap_sentiment", "description": "Train XGBoost + SHAP explainability", "command": f"{py} -m pipelines.training.train_xgb_shap_sentiment --allow_pseudo_fallback"},
            {"step_id": "build_user_personas", "description": "Build user personas", "command": f"{py} -m pipelines.training.build_user_personas"},
            {"step_id": "train_whatsapp_gru_mood_swings", "description": "Train GRU mood swing model", "command": f"{py} -m pipelines.training.train_whatsapp_gru_mood_swings"},
        ]


def run_ordered_pipeline_action(start_from: str | None = None, stop_after: str | None = None, dry_run: bool = False) -> tuple[bool, str]:
    cmd = [sys.executable, "run_pipeline.py"]
    if dry_run:
        cmd.append("--dry-run")
    if start_from:
        cmd.extend(["--start-from", start_from])
    if stop_after:
        cmd.extend(["--stop-after", stop_after])
    try:
        proc = subprocess.run(
            cmd,
            cwd=str(BASE_DIR),
            capture_output=True,
            text=True,
            check=False,
        )
    except Exception as exc:
        return False, f"Failed to launch pipeline command: {exc}"

    stdout = (proc.stdout or "").strip()
    stderr = (proc.stderr or "").strip()
    combined = "\n".join(part for part in [stdout, stderr] if part)
    if not combined:
        combined = "No output captured from pipeline process."
    return proc.returncode == 0, combined


def infer_auto_pipeline_start_step() -> str | None:
    gnn_required = [
        OUTPUT_DIR / "user_behaviour_scores.csv",
        OUTPUT_DIR / "user_feature_importance_global.csv",
        OUTPUT_DIR / "user_feature_importance_per_user.csv",
        USER_EMBEDDINGS_PATH,
    ]
    xgb_required = [
        XGB_PREDICTIONS_PATH,
        XGB_ARTIFACT_DIR / "xgb_target_report.csv",
        XGB_ARTIFACT_DIR / "xgb_embedding_feature_importance.csv",
    ]
    persona_required = [
        PERSONA_TABLE_PATH,
        PERSONA_PROFILE_PATH,
        PERSONA_IMPORTANCE_PATH,
    ]
    gru_required = [
        GRU_MOOD_SWING_SUMMARY_PATH,
        GRU_MOOD_TRAINING_REPORT_PATH,
    ]

    if any(not p.exists() for p in gnn_required):
        return "train_user_behavior_gnn"
    if any(not p.exists() for p in xgb_required):
        return "train_xgb_shap_sentiment"
    if any(not p.exists() for p in persona_required):
        return "build_user_personas"
    if any(not p.exists() for p in gru_required):
        return "train_whatsapp_gru_mood_swings"
    return None


def maybe_run_pipeline_automatically() -> None:
    auto_on = os.getenv("MAYA_AUTO_RUN_PIPELINE", "1").strip().lower() in {"1", "true", "yes"}
    if not auto_on:
        return
    if st.session_state.get("_auto_pipeline_checked", False):
        return
    st.session_state["_auto_pipeline_checked"] = True

    start_step = infer_auto_pipeline_start_step()
    if not start_step:
        return

    with st.spinner(f"Auto-running pipeline from '{start_step}' to generate missing analysis artifacts..."):
        ok, logs = run_ordered_pipeline_action(start_from=start_step, dry_run=False)

    st.session_state["pipeline_last_result"] = {
        "ok": bool(ok),
        "logs": logs,
        "dry_run": False,
        "start_from": start_step,
        "stop_after": "(last)",
        "ran_at": pd.Timestamp.now(tz="Asia/Kolkata").strftime("%Y-%m-%d %H:%M:%S %Z"),
        "auto": True,
    }

    if ok:
        # Clear cached loaders once so the dashboard picks up newly generated artifacts.
        st.cache_data.clear()
        st.rerun()
    else:
        st.warning("Automatic pipeline run failed. Set `MAYA_AUTO_RUN_PIPELINE=0` to disable auto-run and inspect logs in terminal.")


def main() -> None:
    style_app()
    maybe_run_pipeline_automatically()

    st.title("User Behavior Intelligence Dashboard")
    st.caption("User-level feature importance and sentiment insights from your trained GNN outputs.")

    page = st.sidebar.radio(
        "Page",
        ["Global Insights", "Per-User Analysis", "RAG Roadmap Signals", "Persona Analysis", "WhatsApp Sentiment"],
    )

    if page == "WhatsApp Sentiment":
        wa = load_whatsapp_sentiment_messages()
        user_directory = load_user_directory()

        if wa.empty:
            st.info("No WhatsApp sentiment data found. Ensure sentiment_scores.csv exists with message rows.")
            return

        name_map: Dict[int, str] = {int(uid): f"User ({int(uid)})" for uid in sorted(wa["user_id"].unique().tolist())}
        if not user_directory.empty:
            for _, r in user_directory.iterrows():
                uid = int(r["user_id"])
                if uid in name_map:
                    name_map[uid] = str(r["display_name"])

        st.subheader("WhatsApp User Sentiment Overview")
        m1, m2, m3 = st.columns(3)
        m1.metric("Users", f"{wa['user_id'].nunique()}")
        m2.metric("Messages", f"{len(wa)}")
        m3.metric("Global Avg Sentiment", f"{wa['sentiment_score'].mean():.3f}")

        st.subheader("Sentiment Quality Monitor")
        st.caption("Tracks sentiment confidence and label drift so weak/uncertain scoring is visible in the UI.")

        wa_quality = wa.copy()
        wa_quality["sentiment_score"] = pd.to_numeric(wa_quality.get("sentiment_score"), errors="coerce").fillna(0.0).clip(-1.0, 1.0)
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
        q1.metric("Avg Confidence (14d)", f"{recent_conf:.1%}", delta=f"{conf_drift:+.1%}" if pd.notna(conf_drift) else None)
        q2.metric("Low-Confidence Rate", f"{low_conf_rate:.1%}")
        q3.metric("Negative Share (14d)", f"{recent_neg_ratio:.1%}", delta=f"{neg_drift:+.1%}" if pd.notna(neg_drift) else None)
        q4.metric("Uncertain Messages", f"{int((wa_quality['confidence_proxy'] < 0.20).sum()):,}")

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

        uncertain_cols = [c for c in ["created_at", "user_id", "sentiment_score", "sentiment_label", "message"] if c in wa_quality.columns]
        uncertain_rows = wa_quality.sort_values("confidence_proxy", ascending=True).head(25).copy()
        if not uncertain_rows.empty and uncertain_cols:
            st.caption("Lowest-confidence samples to inspect model quality and edge cases.")
            st.dataframe(uncertain_rows[uncertain_cols], width="stretch", height=260)

        st.subheader("Mood Swing Model (GRU)")
        st.caption("Train a GRU on per-user sentiment sequences to estimate mood volatility over time.")
        action_col, report_col = st.columns([1, 2])
        with action_col:
            if st.button("Train GRU Mood Swing Model", use_container_width=True):
                with st.spinner("Training GRU mood model on WhatsApp sentiment timeline..."):
                    ok, logs = run_gru_mood_training_action()
                load_gru_mood_swing_summary.clear()
                load_gru_mood_training_report.clear()
                if ok:
                    st.success("GRU training completed. Mood-swing artifacts were updated.")
                else:
                    st.error("GRU training failed. Check logs below.")
                st.code(logs[-4000:])

        with report_col:
            report = load_gru_mood_training_report()
            if not report.empty:
                rr = report.iloc[0]
                g1, g2, g3 = st.columns(3)
                g1.metric("Eligible Users", f"{int(pd.to_numeric(rr.get('eligible_users', 0), errors='coerce') or 0)}")
                g2.metric("Train Samples", f"{int(pd.to_numeric(rr.get('train_samples', 0), errors='coerce') or 0)}")
                g3.metric("Validation MSE", f"{float(pd.to_numeric(rr.get('val_mse', 0.0), errors='coerce') or 0.0):.4f}")
            else:
                st.info("No GRU training report found yet. Run the training action to generate it.")

        mood_summary = load_gru_mood_swing_summary()
        if not mood_summary.empty:
            mood_view = mood_summary.copy()
            mood_view["user"] = mood_view["user_id"].map(name_map).fillna("User (" + mood_view["user_id"].astype(str) + ")")
            mood_view["user_short"] = mood_view["user"].apply(lambda s: shorten_user_label(s, 26))

            fig_mood = px.bar(
                mood_view.sort_values("mood_swing_index", ascending=True).tail(min(20, len(mood_view))),
                x="mood_swing_index",
                y="user_short",
                orientation="h",
                color="risk_flag",
                color_discrete_map=RISK_COLORS,
                hover_data=["messages", "actual_volatility", "predicted_volatility", "prediction_mae", "trend"],
                title="Top Users by GRU Mood Swing Index",
            )
            style_chart(fig_mood, height=460, x_title="Mood Swing Index", y_title="User")
            st.plotly_chart(fig_mood, width="stretch")

            cols = [
                c
                for c in [
                    "user",
                    "user_id",
                    "messages",
                    "mood_swing_index",
                    "risk_flag",
                    "trend",
                    "prediction_mae",
                    "recommendation",
                ]
                if c in mood_view.columns
            ]
            st.dataframe(
                mood_view[cols].sort_values("mood_swing_index", ascending=False),
                width="stretch",
                height=320,
            )
        else:
            st.info("No GRU mood-swing summary found yet. Train the model to populate user-level mood analysis.")

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

        st.subheader("Per-User WhatsApp Sentiment Trend")
        users = sorted(wa["user_id"].unique().tolist())
        selected_uid = st.selectbox("Select User", users, format_func=lambda uid: name_map.get(int(uid), f"User ({uid})"))
        u = wa[wa["user_id"] == int(selected_uid)].copy().sort_values("created_at")

        tleft, tright = st.columns(2)
        with tleft:
            if u["created_at"].notna().any():
                fig_t = px.line(
                    u.dropna(subset=["created_at"]),
                    x="created_at",
                    y="sentiment_score",
                    markers=True,
                    title="Message Sentiment Over Time",
                )
                fig_t.add_hline(y=0.1, line_dash="dash")
                fig_t.add_hline(y=-0.1, line_dash="dash")
                style_chart(fig_t, height=420, x_title="Timestamp", y_title="Sentiment Score")
                st.plotly_chart(fig_t, width="stretch")
            else:
                st.info("No timestamped messages for selected user.")

        with tright:
            fig_h = px.histogram(
                u,
                x="sentiment_score",
                nbins=20,
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

    if page == "RAG Roadmap Signals":
        sentiment_df = load_sentiment_table()
        roadmap = build_rag_roadmap_signals(sentiment_df, recent_days=30, top_k=14)
        st.subheader("RAG Roadmap Signals")
        st.caption("Prioritize capabilities with strong demand, rising trend, and higher dissatisfaction.")

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

            st.markdown("### Prediction Snapshot")
            with st.container(border=True):
                s1, s2, s3 = st.columns(3)
                s1.metric("Total Users", f"{total_users:,}")
                s2.metric("Positive Predictions", f"{positive_count:,}")
                s3.metric("Negative Predictions", f"{negative_count:,}")

                improve_view = pred_summary.copy()
                improve_view["confidence"] = pd.to_numeric(improve_view.get("confidence"), errors="coerce")
                if improve_view["confidence"].isna().all():
                    improve_view["confidence"] = (2.0 * (improve_view["pred_prob_positive"] - 0.5).abs()).clip(0.0, 1.0)
                else:
                    improve_view["confidence"] = improve_view["confidence"].fillna(
                        (2.0 * (improve_view["pred_prob_positive"] - 0.5).abs()).clip(0.0, 1.0)
                    )

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

                    if 0.45 <= prob_pos <= 0.55:
                        return "Borderline score; add more labeled feedback samples for this user."
                    if confidence < 0.60:
                        return "Low confidence; gather recent examples and retrain with fresher labels."
                    if pred_class == "negative" and prob_pos < 0.25:
                        return "Strong negative signal; review message context and add nuanced sentiment labels."
                    if pred_class == "positive" and prob_pos > 0.80:
                        return "Stable prediction; keep monitoring for drift with periodic label audits."
                    return "Improve feature coverage using richer behavioral and interaction-level signals."

                improve_view["improvement_action"] = improve_view.apply(recommendation_for_row, axis=1)
                show_cols = [
                    c
                    for c in ["user", "user_id", "predicted_class", "pred_prob_positive", "confidence", "improvement_action"]
                    if c in improve_view.columns
                ]
                st.dataframe(
                    improve_view[show_cols].sort_values(["confidence", "pred_prob_positive"], ascending=[True, True]),
                    width="stretch",
                    height=330,
                )
        else:
            st.info("Prediction snapshot card is unavailable because per-user XGBoost predictions were not found.")

        with st.expander("Advanced: Embedding Model Predictions", expanded=False):
            st.caption("Internal model outputs from embedding-based classifier.")
            pred_df = load_xgb_user_predictions()
            report = load_xgb_target_report()

            if not report.empty:
                r = report.iloc[0]
                t1, t2, t3 = st.columns(3)
                t1.metric("Target Source", str(r.get("target_source", "N/A")).replace("_", " ").title())
                t2.metric("Accuracy", f"{float(r.get('accuracy', np.nan)):.3f}" if pd.notna(r.get("accuracy", np.nan)) else "N/A")
                t3.metric("AUC", f"{float(r.get('auc', np.nan)):.3f}" if pd.notna(r.get("auc", np.nan)) else "N/A")

            if pred_df.empty:
                st.info("No xgb_user_predictions.csv found. Re-run train_xgb_shap_sentiment.py to generate per-user predictions.")
            else:
                names = load_user_directory()
                pred_view = pred_df.copy()
                if not names.empty:
                    pred_view = pred_view.merge(names, on="user_id", how="left")
                    pred_view["user"] = pred_view["display_name"].fillna("User (" + pred_view["user_id"].astype(str) + ")")
                else:
                    pred_view["user"] = "User (" + pred_view["user_id"].astype(str) + ")"
                pred_view["pred_prob_negative"] = 1.0 - pred_view["pred_prob_positive"].astype(float)
                pred_view = pred_view.sort_values("pred_prob_positive", ascending=False)

                p1, p2, p3 = st.columns(3)
                p1.metric("Users Scored", f"{len(pred_view):,}")
                p2.metric("Predicted Positive Rate", f"{(pred_view['pred_prob_positive'] >= 0.5).mean():.1%}")
                p3.metric("Avg Positive Probability", f"{pred_view['pred_prob_positive'].mean():.3f}")

                all_rows = pred_view.sort_values("pred_prob_positive", ascending=True).copy()
                prob_long = all_rows.melt(
                    id_vars=["user", "user_id", "predicted_class", "confidence"],
                    value_vars=["pred_prob_positive", "pred_prob_negative"],
                    var_name="probability_type",
                    value_name="probability",
                )
                prob_long["probability_type"] = prob_long["probability_type"].replace(
                    {
                        "pred_prob_positive": "Positive Probability",
                        "pred_prob_negative": "Negative Probability",
                    }
                )
                chart_h = int(min(max(500, 18 * len(all_rows) + 120), 2200))
                fig_pred = px.bar(
                    prob_long,
                    x="probability",
                    y="user",
                    color="probability_type",
                    barmode="group",
                    orientation="h",
                    title="All Users: Positive vs Negative Probability",
                    hover_data=["user_id", "predicted_class", "confidence"],
                    color_discrete_map={
                        "Positive Probability": "#2E8B57",
                        "Negative Probability": "#B2413E",
                    },
                )
                style_chart(fig_pred, height=chart_h, x_title="Probability", y_title="User")
                fig_pred.update_xaxes(range=[0, 1])
                st.plotly_chart(fig_pred, width="stretch")

                out_cols = [
                    c
                    for c in [
                        "user",
                        "user_id",
                        "predicted_class",
                        "pred_prob_positive",
                        "pred_prob_negative",
                        "confidence",
                        "target",
                    ]
                    if c in pred_view.columns
                ]
                st.dataframe(pred_view[out_cols], width="stretch", height=520)
        return

    if page == "Persona Analysis":
        persona_table, persona_profiles, persona_importance = load_persona_outputs()
        persona_user_shap = load_persona_user_shap()
        persona_user_directory = load_user_directory()
        dissatisfaction_df = load_user_dissatisfaction_flags()

        if persona_table.empty or persona_profiles.empty:
            st.info("Persona outputs not found. Run build_user_personas.py first.")
            return

        m1, m2, m3 = st.columns(3)
        m1.metric("Users With Persona", f"{persona_table['user_id'].nunique()}")
        m2.metric("Total Personas", f"{persona_profiles['persona_label'].nunique()}")
        m3.metric("Largest Persona", str(persona_profiles.sort_values('users', ascending=False).iloc[0]['persona_label']))

        left, right = st.columns(2)
        with left:
            st.subheader("Persona Distribution")
            dist = persona_profiles[["persona_label", "users"]].copy().sort_values("users", ascending=False)
            fig_dist = px.bar(dist, x="persona_label", y="users", title="Users Per Persona")
            style_chart(fig_dist, height=420, x_title="Persona", y_title="Users", rotate_x=True)
            st.plotly_chart(fig_dist, width="stretch")

        with right:
            st.subheader("Persona Sentiment Profile")
            if "avg_sentiment" in persona_profiles.columns:
                fig_sent = px.bar(
                    persona_profiles.sort_values("avg_sentiment", ascending=False),
                    x="persona_label",
                    y="avg_sentiment",
                    color="avg_sentiment",
                    color_continuous_scale=SENTIMENT_DIVERGING_SCALE,
                    color_continuous_midpoint=0.0,
                    title="Average Sentiment By Persona",
                )
                style_chart(fig_sent, height=420, x_title="Persona", y_title="Avg Sentiment", rotate_x=True)
                st.plotly_chart(fig_sent, width="stretch")

        st.subheader("Persona-Level Summaries")
        st.dataframe(persona_profiles.sort_values("users", ascending=False), width="stretch", height=300)

        st.subheader("Global Feature Importance (Behavior Only)")
        if persona_importance.empty:
            st.info("No persona feature importance file found.")
        else:
            geo_noise = ["latitude", "longitude", "timezone", "country"]
            keep_mask = ~persona_importance["feature"].astype(str).str.contains("|".join(geo_noise), case=False, regex=True)
            imp_clean = persona_importance[keep_mask].copy()
            fig_fi = px.bar(
                imp_clean.head(15).sort_values("importance", ascending=True),
                x="importance",
                y="feature",
                orientation="h",
                title="Global Persona Drivers (Geographic Noise Removed)",
            )
            style_chart(fig_fi, height=500, x_title="Importance", y_title="Feature")
            st.plotly_chart(fig_fi, width="stretch")

        st.subheader("t-SNE of GraphSAGE Embeddings By Persona")
        tsne_df = build_tsne_persona(persona_table)
        if tsne_df.empty:
            st.info("Not enough embedding/persona data to build t-SNE plot.")
        else:
            fig_tsne = px.scatter(
                tsne_df,
                x="tsne_x",
                y="tsne_y",
                color="persona_label",
                color_discrete_sequence=PERSONA_COLORS,
                hover_data=["user_id"],
                title="User Embedding Clusters (t-SNE)",
            )
            fig_tsne.update_traces(marker=dict(size=9, opacity=0.9, line=dict(width=0.6, color="#ffffff")))
            style_chart(fig_tsne, height=520, x_title="t-SNE 1", y_title="t-SNE 2")
            st.plotly_chart(fig_tsne, width="stretch")

        st.subheader("Persona SHAP Summary Plot")
        if PERSONA_SHAP_PLOT_PATH.exists():
            st.image(str(PERSONA_SHAP_PLOT_PATH), width="stretch")
        else:
            st.info("No persona SHAP plot found. Re-run build_user_personas.py to generate persona_shap_summary.png.")

        st.subheader("Top 3 Reasons By Persona")
        reason_summary = summarize_persona_reasons(persona_table)
        if reason_summary.empty:
            st.info("No persona reason summaries available.")
        else:
            st.dataframe(reason_summary, width="stretch", height=240)

        st.subheader("User Persona Table")
        if not persona_user_directory.empty and "user_id" in persona_table.columns:
            view_names = persona_user_directory[["user_id", "display_name"]].drop_duplicates("user_id")
            persona_table = persona_table.merge(view_names, on="user_id", how="left")
            persona_table["user"] = persona_table["display_name"].fillna("User")
            persona_table.loc[persona_table["display_name"].isna(), "user"] = (
                "User (" + persona_table["user_id"].astype(str) + ")"
            )
            persona_table = persona_table.drop(columns=["display_name"])
            ordered_cols = [
                "user",
                "user_id",
                "persona_label",
                "top_reason_1",
                "top_reason_2",
                "top_reason_3",
            ]
            persona_table = persona_table[[c for c in ordered_cols if c in persona_table.columns]]

        if not dissatisfaction_df.empty and "user_id" in persona_table.columns:
            persona_table = persona_table.merge(
                dissatisfaction_df[["user_id", "dissatisfaction_flag", "dissatisfaction_score", "dissatisfaction_reason"]],
                on="user_id",
                how="left",
            )

        st.subheader("Dissatisfaction Risk Distribution")
        if not dissatisfaction_df.empty:
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
                title="Users By Dissatisfaction Risk",
            )
            style_chart(fig_risk, height=360, x_title="Risk", y_title="Users")
            st.plotly_chart(fig_risk, width="stretch")
        else:
            st.info("Dissatisfaction scores unavailable (sentiment_scores.csv not found or missing required columns).")

        persona_labels = sorted(persona_table["persona_label"].dropna().astype(str).unique().tolist())
        selected_persona = st.selectbox("Filter By Persona", ["All"] + persona_labels)
        sentiment_filter = st.sidebar.selectbox("Persona Sentiment Filter", ["All", "Frustrated", "Neutral", "Satisfied"], key="persona_sent_filter")
        activity_filter = st.sidebar.selectbox("Persona Activity Filter", ["All", "Highly Active", "Low Activity"], key="persona_act_filter")
        view = persona_table.copy()
        if selected_persona != "All":
            view = view[view["persona_label"].astype(str) == selected_persona]
        if sentiment_filter != "All":
            view = view[view["persona_label"].astype(str).str.contains(sentiment_filter, case=False, na=False)]
        if activity_filter != "All":
            view = view[view["persona_label"].astype(str).str.contains(activity_filter, case=False, na=False)]
        st.dataframe(view.sort_values("user_id"), width="stretch", height=420)

        st.subheader("Per-User Exact Feature Analysis")
        if persona_user_shap.empty:
            st.info("No per-user SHAP contribution file found. Re-run build_user_personas.py.")
        else:
            users = sorted(persona_user_shap["user_id"].dropna().astype(int).unique().tolist())
            selected_uid = st.selectbox("Select User For Exact Analysis", users)
            u_df = persona_user_shap[persona_user_shap["user_id"] == selected_uid].copy()
            u_df = u_df.sort_values("abs_shap", ascending=False)

            u_meta = persona_table[persona_table["user_id"] == selected_uid].head(1)
            if not u_meta.empty:
                user_name = u_meta["user"].iloc[0] if "user" in u_meta.columns else f"User ({selected_uid})"
                risk = u_meta["dissatisfaction_flag"].iloc[0] if "dissatisfaction_flag" in u_meta.columns else "N/A"
                st.caption(f"User: {user_name} | Persona: {u_meta['persona_label'].iloc[0]} | Dissatisfaction Risk: {risk}")
                if "dissatisfaction_reason" in u_meta.columns and pd.notna(u_meta["dissatisfaction_reason"].iloc[0]):
                    st.caption(f"Risk Driver: {u_meta['dissatisfaction_reason'].iloc[0]}")

            fig_u = px.bar(
                u_df.head(12).sort_values("abs_shap", ascending=True),
                x="abs_shap",
                y="feature",
                orientation="h",
                title=f"Top Local Drivers For User {selected_uid}",
            )
            style_chart(fig_u, height=460, x_title="|SHAP| Contribution", y_title="Feature")
            st.plotly_chart(fig_u, width="stretch")
            st.dataframe(u_df.head(20), width="stretch", height=320)
        return

    scores, global_imp, per_user_imp = load_outputs()
    user_directory = load_user_directory()
    sentiment_df = load_sentiment_table()

    if scores.empty:
        st.error("No GNN output files found in gnn_outputs.")
        st.stop()

    score_users = sorted(scores["user_id"].dropna().astype(int).tolist())

    display_map: Dict[int, str] = {u: f"User ({u})" for u in score_users}
    if not user_directory.empty:
        for _, row in user_directory.iterrows():
            uid = int(row["user_id"])
            if uid in display_map:
                display_map[uid] = str(row["display_name"])

    if "selected_user_per_user" not in st.session_state:
        st.session_state["selected_user_per_user"] = int(score_users[0]) if score_users else -1
    if "refresh_by_user" not in st.session_state:
        st.session_state["refresh_by_user"] = {}
    refresh_by_user = st.session_state["refresh_by_user"]

    if page == "Global Insights":
        global_tasks = build_task_importance(sentiment_df, user_id=None, top_k=20)
        global_feature_focus = build_feature_focus_summary(sentiment_df, top_k=12)
        global_statements = build_representative_statements(sentiment_df, user_id=None, top_k=12)

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
                sent_counts = sentiment_df.groupby("sentiment").size().reset_index(name="count")
                fig_pie = px.pie(
                    sent_counts,
                    names="sentiment",
                    values="count",
                    hole=0.45,
                    color="sentiment",
                    color_discrete_map=SENTIMENT_COLORS,
                )
                style_chart(fig_pie, height=420, kind="pie")
                st.plotly_chart(fig_pie, width="stretch")

        with right:
            st.subheader("Global Sentiment Over Time")
            time_df = sentiment_df.dropna(subset=["created_at"]).copy()
            if time_df.empty:
                st.info("No timestamped sentiment rows available.")
            else:
                time_df["date"] = time_df["created_at"].dt.date
                daily = time_df.groupby("date", as_index=False).agg(avg_polarity=("polarity", "mean"))
                fig_time = px.line(daily, x="date", y="avg_polarity", markers=True)
                fig_time.add_hline(y=0.1, line_dash="dash")
                fig_time.add_hline(y=-0.1, line_dash="dash")
                style_chart(fig_time, height=420, x_title="Date", y_title="Average Polarity")
                st.plotly_chart(fig_time, width="stretch")

        st.subheader("Global Most Requested Intents")
        if global_tasks.empty:
            st.info("No global task ranking available.")
        else:
            fig_global_chat = px.bar(
                global_tasks.head(15).sort_values("importance", ascending=True),
                x="importance",
                y="task",
                orientation="h",
                title="Global Intent Importance",
            )
            style_chart(fig_global_chat, height=460, x_title="Importance", y_title="Task")
            st.plotly_chart(fig_global_chat, width="stretch")
            st.dataframe(global_tasks[["task", "mentions", "avg_polarity", "sample_request"]].head(12), width="stretch", height=280)

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
            st.dataframe(focus_table, width="stretch", height=300)
            st.caption("Use high-mention clusters as top RAG coverage priorities, then inspect sample request phrases for intent granularity.")

        st.subheader("Representative Global Statements")
        if global_statements.empty:
            st.info("No global contextual statements found.")
        else:
            st.dataframe(global_statements, width="stretch", height=320)

        st.subheader("Global GNN Feature Importance")
        top_global = global_imp.head(12).sort_values("importance", ascending=True)
        fig_global = px.bar(
            top_global,
            x="importance",
            y="feature",
            orientation="h",
            title="Global GNN Model Features",
        )
        style_chart(fig_global, height=460, x_title="Importance", y_title="Feature")
        st.plotly_chart(fig_global, width="stretch")

    else:
        st.subheader("Per-User Controls")
        control_left, control_mid, control_right = st.columns([2.5, 2.2, 1.0])
        with control_left:
            search_query = st.text_input(
                "Search User (Name or ID)",
                value=st.session_state.get("per_user_search_query", ""),
                placeholder="Try: tushi, joshi, or 123",
                key="per_user_search_query",
            ).strip().lower()
        matching_users = [
            uid
            for uid in score_users
            if not search_query
            or search_query in display_map.get(uid, "").lower()
            or search_query in str(uid)
        ]
        if not matching_users:
            st.warning("No users matched your search. Showing full user list.")
            matching_users = score_users

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
                fig_radar = go.Figure()
                fig_radar.add_trace(
                    go.Scatterpolar(
                        r=r,
                        theta=theta,
                        fill="toself",
                        line=dict(color=ACCENT_PRIMARY, width=3),
                        marker=dict(size=6, color=ACCENT_PRIMARY),
                        name="Latest 5 Interactions",
                    )
                )
                fig_radar.update_layout(
                    height=380,
                    polar=dict(
                        bgcolor=CHART_PLOT_BG,
                        radialaxis=dict(visible=True, range=[0, 100], gridcolor=GRID_COLOR, tickfont=dict(size=11)),
                        angularaxis=dict(gridcolor=GRID_COLOR, tickfont=dict(size=12)),
                    ),
                    showlegend=False,
                    paper_bgcolor=CHART_PAPER_BG,
                    margin=dict(l=44, r=32, t=40, b=30),
                )
                st.plotly_chart(fig_radar, width="stretch")
                st.caption("Scores summarize the latest 5 user interactions.")

        timeline_col, hri_col = st.columns(2)
        with timeline_col:
            st.markdown("#### Interaction Timeline")
            timeline = build_response_sentiment_timeline(selected_user, user_sent)
            if timeline.empty:
                st.info("Not enough user→assistant turns to compute response-time timeline.")
            else:
                fig_timeline = go.Figure()
                fig_timeline.add_trace(
                    go.Scatter(
                        x=timeline["created_at"],
                        y=timeline["response_time_sec"],
                        name="Response Time (sec)",
                        mode="lines+markers",
                        line=dict(color=ACCENT_PRIMARY, width=2.5),
                        marker=dict(size=6),
                    )
                )
                fig_timeline.add_trace(
                    go.Scatter(
                        x=timeline["created_at"],
                        y=timeline["polarity"],
                        name="Sentiment Score",
                        mode="lines+markers",
                        yaxis="y2",
                        line=dict(color="#B2413E", width=2),
                        marker=dict(size=5),
                    )
                )
                style_chart(fig_timeline, height=420, x_title="Timestamp", y_title="Response Time (sec)")
                fig_timeline.update_layout(
                    yaxis2=dict(
                        title="Sentiment Score",
                        overlaying="y",
                        side="right",
                        range=[-1.0, 1.0],
                        showgrid=False,
                        zeroline=True,
                    ),
                    legend=dict(orientation="h", y=1.08, x=0.01),
                )
                st.plotly_chart(fig_timeline, width="stretch")
                st.caption("Dual-axis view helps catch slow-response + negative-sentiment trends early.")

        with hri_col:
            st.markdown("#### HRI Metrics")
            modality_df, physical_df, proxy_used = build_hri_metrics(selected_user, user_sent)
            h1, h2 = st.columns(2)
            with h1:
                fig_modality = px.bar(
                    modality_df,
                    x="metric",
                    y="value",
                    color="metric",
                    color_discrete_sequence=[ACCENT_PRIMARY, "#EA8C55"],
                    title="Modality Usage",
                )
                style_chart(fig_modality, height=390, x_title="Modality", y_title="Interactions", rotate_x=False)
                st.plotly_chart(fig_modality, width="stretch")
            with h2:
                fig_physical = px.bar(
                    physical_df,
                    x="metric",
                    y="value",
                    color="metric",
                    color_discrete_sequence=["#2A9D8F", "#8A5A44"],
                    title="Physical Engagement",
                )
                fig_physical.update_yaxes(range=[0, 100])
                style_chart(fig_physical, height=390, x_title="Signal", y_title="Score")
                st.plotly_chart(fig_physical, width="stretch")
            if proxy_used:
                st.caption("Physical engagement scores are proxy-derived because explicit distance/posture sensor columns are not available in current datasets.")

        st.divider()
        st.markdown("### Drivers And Requests")
        left, right = st.columns(2)
        with left:
            st.subheader("Top Features For Selected User")
            if user_imp.empty:
                st.info("No feature-importance rows found for this user.")
            else:
                user_imp_display = user_imp.copy()
                user_imp_display["feature_readable"] = user_imp_display["feature"].apply(humanize_feature_name)
                st.caption("Showing non-geographic features with non-zero contribution. Hover each bar to see the original raw feature key.")
                fig_user = px.bar(
                    user_imp_display.sort_values("importance", ascending=True),
                    x="importance",
                    y="feature_readable",
                    orientation="h",
                    hover_data=["feature"],
                    title="Per-user Feature Importance",
                )
                style_chart(fig_user, height=460, x_title="Importance", y_title="Feature")
                st.plotly_chart(fig_user, width="stretch")

        with right:
            st.subheader("Most Requested Intents By User")
            if task_imp.empty:
                st.info("No task-like requests found for this user.")
            else:
                fig_chat = px.bar(
                    task_imp.head(12).sort_values("importance", ascending=True),
                    x="importance",
                    y="task",
                    orientation="h",
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
                sent_counts = user_sent.groupby("sentiment").size().reset_index(name="count")
                fig_pie = px.pie(
                    sent_counts,
                    names="sentiment",
                    values="count",
                    hole=0.45,
                    color="sentiment",
                    color_discrete_map=SENTIMENT_COLORS,
                )
                style_chart(fig_pie, height=400, kind="pie")
                st.plotly_chart(fig_pie, width="stretch")

            with d2:
                time_df = user_sent.dropna(subset=["created_at"]).sort_values("created_at")
                if not time_df.empty:
                    fig_time = px.line(
                        time_df,
                        x="created_at",
                        y="polarity",
                        color="source",
                        markers=True,
                        title="Sentiment Over Time",
                    )
                    fig_time.add_hline(y=0.1, line_dash="dash")
                    fig_time.add_hline(y=-0.1, line_dash="dash")
                    style_chart(fig_time, height=400, x_title="Timestamp", y_title="Sentiment Score")
                    st.plotly_chart(fig_time, width="stretch")

            show_cols = ["created_at", "source", "message", "polarity", "subjectivity", "sentiment"]
            st.dataframe(user_sent[show_cols].sort_values("created_at", ascending=False).head(25), width="stretch", height=360)


if __name__ == "__main__":
    main()
