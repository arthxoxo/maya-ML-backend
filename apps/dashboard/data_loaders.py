"""
Data loading, caching, and building functions for the Maya dashboard.
All @st.cache_data functions and data transformation utilities live here.
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
import streamlit as st
from sklearn.manifold import TSNE

from apps.dashboard.shared import (
    BASE_DIR, OUTPUT_DIR, PREPROCESSED_DIR, SECRET_DATA_DIR,
    FLINK_ENGINEERED_DIR, EMBEDDINGS_ARTIFACT_DIR, XGB_ARTIFACT_DIR,
    PERSONA_ARTIFACT_DIR, SENTIMENT_ARTIFACT_DIR,
    XGB_SHAP_IMPORTANCE_PATH, XGB_PREDICTIONS_PATH, XGB_MODEL_PATH_CANDIDATES,
    EMBEDDING_LABELS_PATH, USER_EMBEDDINGS_PATH,
    PERSONA_TABLE_PATH, PERSONA_PROFILE_PATH, PERSONA_IMPORTANCE_PATH,
    PERSONA_USER_SHAP_PATH, PERSONA_SHAP_PLOT_PATH,
    SENTIMENT_SCORES_PATH, GRU_MOOD_SWING_SUMMARY_PATH, GRU_MOOD_TRAINING_REPORT_PATH,
    SESSIONS_SOURCE_PATH, RAW_USERS_PATH, RAW_SESSIONS_PATH, RAW_MESSAGES_PATH,
    HF_SENTIMENT_MODEL, HF_IRONY_MODEL,
    REDIS_KEY_PREFIX, GEO_NOISE_PATTERN,
    SENTIMENT_COLORS, ACCENT_PRIMARY, PERSONA_COLORS, RISK_COLORS,
    STOPWORDS, FILLER_WORDS, ACTION_VERBS,
    TASK_PATTERNS, CANONICAL_INTENT_PATTERNS, FEATURE_FOCUS_PATTERNS, HUMANOID_TOOL_PATTERNS,
    _first_existing_path, get_redis_client, _empty_df, load_df_from_redis,
    heuristic_sentiment_fallback,
    humanize_feature_name, simplify_persona_label, shorten_user_label,
    remove_geographic_noise, remove_non_actionable_feature_noise,
)

@st.cache_data(show_spinner=False)
def load_outputs(refresh_nonce: str | None = None) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    _ = refresh_nonce
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

    scores_path = OUTPUT_DIR / "user_behaviour_scores.csv"
    global_path = OUTPUT_DIR / "user_feature_importance_global.csv"
    per_user_path = OUTPUT_DIR / "user_feature_importance_per_user.csv"

    if not (scores_path.exists() and global_path.exists() and per_user_path.exists()):
        return (
            _empty_df(["user_id", "engagement_score", "pred_high_engagement_prob"]),
            _empty_df(["feature", "importance"]),
            _empty_df(["user_id", "rank", "feature", "importance"]),
        )

    scores = pd.read_csv(scores_path)
    global_imp = pd.read_csv(global_path)
    per_user_imp = pd.read_csv(per_user_path)
    return scores, global_imp, per_user_imp


def gnn_output_file_status() -> tuple[bool, list[str]]:
    expected = [
        OUTPUT_DIR / "user_behaviour_scores.csv",
        OUTPUT_DIR / "user_feature_importance_global.csv",
        OUTPUT_DIR / "user_feature_importance_per_user.csv",
    ]
    existing = [p for p in expected if p.exists()]
    return bool(existing), [p.name for p in existing]


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
    def _clean_name_col(col: str) -> pd.Series:
        if col in users.columns:
            return users[col].fillna("").astype(str).str.strip()
        return pd.Series("", index=users.index, dtype="object")

    users["first_name"] = _clean_name_col("first_name")
    users["last_name"] = _clean_name_col("last_name")
    users["full_name"] = _clean_name_col("full_name")
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
    if p > 0.15:
        return "positive"
    if p < -0.15:
        return "negative"
    return "neutral"


def calibrate_sentiment_labels(
    scores: pd.Series,
    target_neutral_share: float = 0.65,
    min_threshold: float = 0.005,
    max_threshold: float = 0.22,
) -> tuple[pd.Series, float]:
    s = pd.to_numeric(scores, errors="coerce").fillna(0.0).clip(-1.0, 1.0)
    if s.empty:
        return pd.Series(dtype="object"), float(min_threshold)

    target_n = float(np.clip(target_neutral_share, 0.45, 0.85))
    abs_s = s.abs()
    thr = float(abs_s.quantile(target_n))
    thr = float(np.clip(thr, min_threshold, max_threshold))

    labels = np.where(s > thr, "positive", np.where(s < -thr, "negative", "neutral"))
    return pd.Series(labels, index=s.index, dtype="object"), thr


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
    if flat_ratio < 0.80 and neutral_ratio < 0.70:
        return out

    heur = out[text_col].fillna("").astype(str).apply(lambda t: float(heuristic_sentiment_fallback(t)[0]))
    weak_mask = score.abs() < 0.03
    blended = np.where(weak_mask, heur, 0.75 * score + 0.25 * heur)
    out[score_col] = pd.Series(blended, index=out.index, dtype="float64").fillna(0.0).clip(-1.0, 1.0)
    out[label_col] = out[score_col].apply(polarity_label)
    return out


def _emoji_polarity_hint(text: str) -> float:
    s = str(text or "")
    if not s:
        return 0.0

    pos_emojis = {"😀", "😄", "😁", "🙂", "😊", "😍", "🥰", "👍", "🔥", "🎉", "💯", "❤️", "❤"}
    neg_emojis = {"😡", "😠", "😤", "😞", "😔", "😢", "😭", "👎", "💔", "😒", "🙄", "🤬", "⚠️"}

    pos_hits = sum(1 for e in pos_emojis if e in s)
    neg_hits = sum(1 for e in neg_emojis if e in s)
    hint = 0.10 * pos_hits - 0.12 * neg_hits

    exclam = s.count("!")
    if exclam > 0:
        hint *= 1.0 + min(exclam, 4) * 0.06

    return float(np.clip(hint, -0.35, 0.35))


def _intent_polarity_hint(text: str) -> float:
    s = str(text or "").strip().lower()
    if not s:
        return 0.0

    neg_patterns = [
        r"\bnot\s+working\b",
        r"\bdoesn'?t\s+work\b",
        r"\bcan'?t\b|\bcannot\b|\bunable\b",
        r"\b(error|issue|problem|failed|failure|broken|stuck|refund|angry|upset|frustrated)\b",
        r"\bwhy\s+(is|was|are)\b",
    ]
    pos_patterns = [
        r"\b(thank\s+you|thanks|appreciate)\b",
        r"\b(works\s+well|resolved|fixed|great|awesome|perfect|excellent)\b",
        r"\b(good\s+job|well\s+done)\b",
    ]

    neg_hits = sum(1 for p in neg_patterns if re.search(p, s))
    pos_hits = sum(1 for p in pos_patterns if re.search(p, s))
    hint = 0.18 * pos_hits - 0.22 * neg_hits
    return float(np.clip(hint, -0.45, 0.45))


def strengthen_whatsapp_sentiment(
    df: pd.DataFrame,
    text_col: str = "message",
    score_col: str = "sentiment_score",
    label_col: str = "sentiment_label",
    group_col: str = "user_id",
    time_col: str = "created_at",
    context_window: int = 4,
) -> pd.DataFrame:
    if df.empty or text_col not in df.columns:
        return df

    out = df.copy()
    out[text_col] = out[text_col].fillna("").astype(str)
    if time_col in out.columns:
        out[time_col] = pd.to_datetime(out[time_col], errors="coerce", utc=True)

    base_score = pd.to_numeric(out.get(score_col), errors="coerce").fillna(0.0).clip(-1.0, 1.0)
    heur = out[text_col].apply(heuristic_sentiment_fallback)
    heur_pol = pd.to_numeric(heur.str[0], errors="coerce").fillna(0.0).clip(-1.0, 1.0)
    heur_subj = pd.to_numeric(heur.str[1], errors="coerce").fillna(0.0).clip(0.0, 1.0)
    out["score_model_raw"] = base_score.astype(float)
    out["heuristic_score"] = heur_pol.astype(float)

    lbl = out.get(label_col, "").fillna("").astype(str).str.lower().str.strip()
    weak_or_neutral = base_score.abs().lt(0.12) | lbl.isin(["", "neutral"])

    # Trust heuristic more when baseline signal is weak or neutral.
    blended = np.where(weak_or_neutral, 0.35 * base_score + 0.65 * heur_pol, 0.75 * base_score + 0.25 * heur_pol)

    disagreement = (np.sign(base_score) != np.sign(heur_pol)) & heur_pol.abs().gt(0.35)
    blended = np.where(disagreement, 0.5 * blended + 0.5 * heur_pol, blended)

    emoji_hint = out[text_col].apply(_emoji_polarity_hint)
    intent_hint = out[text_col].apply(_intent_polarity_hint)
    out["score_rule_hint"] = pd.to_numeric(emoji_hint + intent_hint, errors="coerce").fillna(0.0).clip(-1.0, 1.0)
    blended = pd.Series(blended, index=out.index, dtype="float64") + emoji_hint + intent_hint
    blended = blended.clip(-1.0, 1.0)
    pre_context = blended.copy()

    if group_col in out.columns and time_col in out.columns:
        tmp = out[[group_col, time_col]].copy()
        tmp["_score"] = blended.values
        tmp = tmp.sort_values([group_col, time_col], kind="mergesort")

        adjusted_parts: list[pd.Series] = []
        for _, grp in tmp.groupby(group_col, sort=False):
            prior_mean = grp["_score"].shift(1).rolling(window=context_window, min_periods=1).mean().fillna(0.0)
            weak_now = grp["_score"].abs().lt(0.25)
            adjusted = grp["_score"] + np.where(weak_now, 0.18 * prior_mean, 0.08 * prior_mean)
            adjusted_parts.append(pd.Series(adjusted, index=grp.index))

        if adjusted_parts:
            adjusted_all = pd.concat(adjusted_parts).sort_index()
            tmp.loc[adjusted_all.index, "_score"] = adjusted_all
            blended = tmp["_score"].reindex(out.index).fillna(blended).clip(-1.0, 1.0)

    out[score_col] = pd.to_numeric(blended, errors="coerce").fillna(0.0).clip(-1.0, 1.0)
    out[label_col] = out[score_col].apply(polarity_label)
    out["score_pre_context"] = pd.to_numeric(pre_context, errors="coerce").fillna(0.0).clip(-1.0, 1.0)
    out["score_context_adjustment"] = (out[score_col] - out["score_pre_context"]).clip(-1.0, 1.0)

    # Confidence should be high for both strong polarity and clearly neutral intent.
    signal_strength = (0.50 * out[score_col].abs() + 0.25 * heur_pol.abs() + 0.10 * emoji_hint.abs() + 0.15 * intent_hint.abs()).clip(0.0, 1.0)
    neutral_certainty = ((out[score_col].abs() < 0.09) & (heur_pol.abs() < 0.09)).astype(float)
    disagreement_penalty = ((np.sign(base_score) != np.sign(heur_pol)) & heur_pol.abs().gt(0.25)).astype(float) * 0.20
    conf = 0.20 + 0.55 * signal_strength + 0.30 * neutral_certainty + 0.15 * heur_subj - disagreement_penalty
    out["sentiment_confidence"] = pd.to_numeric(conf, errors="coerce").fillna(0.0).clip(0.0, 1.0)

    def _build_debug_flags(row: pd.Series) -> str:
        flags: list[str] = []
        if abs(float(row.get("score_rule_hint", 0.0))) >= 0.06:
            flags.append("rules")
        if abs(float(row.get("score_context_adjustment", 0.0))) >= 0.03:
            flags.append("context")
        if abs(float(row.get("heuristic_score", 0.0))) >= 0.20:
            flags.append("heuristic")
        if abs(float(row.get("score_model_raw", 0.0))) >= 0.20:
            flags.append("model")
        if abs(float(row.get("score_model_raw", 0.0)) - float(row.get("heuristic_score", 0.0))) >= 0.30:
            flags.append("disagreement")
        return ", ".join(flags) if flags else "none"

    out["sentiment_debug_flags"] = out.apply(_build_debug_flags, axis=1)
    return out


def apply_gru_sequence_context(
    df: pd.DataFrame,
    text_col: str = "message",
    score_col: str = "sentiment_score",
    label_col: str = "sentiment_label",
    group_col: str = "user_id",
    time_col: str = "created_at",
) -> pd.DataFrame:
    if df.empty or group_col not in df.columns or time_col not in df.columns:
        return df

    enabled = os.getenv("MAYA_ENABLE_GRU_SENTIMENT_CONTEXT", "1").strip().lower() in {"1", "true", "yes"}
    if not enabled:
        return df

    try:
        import torch
        import torch.nn as nn
    except Exception:
        return df

    out = df.copy()
    out[text_col] = out.get(text_col, "").fillna("").astype(str)
    out[time_col] = pd.to_datetime(out.get(time_col), errors="coerce", utc=True)
    out = out.sort_values([group_col, time_col], kind="mergesort").copy()

    base = pd.to_numeric(out.get(score_col), errors="coerce").fillna(0.0).clip(-1.0, 1.0)
    heur = pd.to_numeric(out.get("heuristic_score"), errors="coerce").fillna(base).clip(-1.0, 1.0)
    emoji_hint = out[text_col].apply(_emoji_polarity_hint)
    intent_hint = out[text_col].apply(_intent_polarity_hint)
    msg_len = out[text_col].str.len().fillna(0).clip(0, 400).astype(float) / 400.0
    punct = out[text_col].str.count(r"[!?]").fillna(0).clip(0, 6).astype(float) / 6.0
    feats = np.column_stack([
        base.to_numpy(dtype=np.float32),
        heur.to_numpy(dtype=np.float32),
        pd.to_numeric(emoji_hint, errors="coerce").fillna(0.0).to_numpy(dtype=np.float32),
        pd.to_numeric(intent_hint, errors="coerce").fillna(0.0).to_numpy(dtype=np.float32),
        (msg_len.to_numpy(dtype=np.float32) * 2.0 - 1.0),
        (punct.to_numpy(dtype=np.float32) * 2.0 - 1.0),
    ])

    lookback = int(os.getenv("MAYA_GRU_LOOKBACK", "8"))
    hidden = int(os.getenv("MAYA_GRU_HIDDEN", "24"))
    epochs = int(os.getenv("MAYA_GRU_EPOCHS", "16"))

    x_train: list[np.ndarray] = []
    y_train: list[float] = []

    def _pad_window(arr: np.ndarray) -> np.ndarray:
        if arr.shape[0] >= lookback:
            return arr[-lookback:, :]
        pad = np.zeros((lookback - arr.shape[0], arr.shape[1]), dtype=np.float32)
        return np.vstack([pad, arr]).astype(np.float32)

    for _, grp in out.groupby(group_col, sort=False):
        idx = grp.index.to_numpy()
        if len(idx) < 2:
            continue
        for t in range(1, len(idx)):
            seq = feats[idx[max(0, t - lookback):t], :]
            x_train.append(_pad_window(seq))
            prior_mean = float(base.loc[idx[max(0, t - lookback):t]].mean())
            target_t = (
                0.50 * float(base.loc[idx[t]])
                + 0.30 * float(heur.loc[idx[t]])
                + 0.12 * float(intent_hint.loc[idx[t]])
                + 0.08 * float(emoji_hint.loc[idx[t]])
                + 0.10 * prior_mean
            )
            y_train.append(float(np.clip(target_t, -1.0, 1.0)))

    if len(x_train) < 32:
        return out.sort_index()

    x_np = np.stack(x_train).astype(np.float32)
    y_np = np.array(y_train, dtype=np.float32).reshape(-1, 1)

    torch.manual_seed(42)

    class _SeqSentimentGRU(nn.Module):
        def __init__(self, in_dim: int, hidden_dim: int):
            super().__init__()
            self.gru = nn.GRU(in_dim, hidden_dim, batch_first=True)
            self.head = nn.Linear(hidden_dim, 1)

        def forward(self, x):
            _, h = self.gru(x)
            return torch.tanh(self.head(h[-1]))

    model = _SeqSentimentGRU(in_dim=x_np.shape[2], hidden_dim=hidden)
    opt = torch.optim.Adam(model.parameters(), lr=0.01)
    loss_fn = nn.MSELoss()

    x_t = torch.from_numpy(x_np)
    y_t = torch.from_numpy(y_np)

    model.train()
    for _ in range(max(4, epochs)):
        opt.zero_grad()
        pred = model(x_t)
        loss = loss_fn(pred, y_t)
        loss.backward()
        opt.step()

    model.eval()
    context_pred = pd.Series(base.to_numpy(dtype=np.float32), index=out.index, dtype="float64")
    with torch.no_grad():
        for _, grp in out.groupby(group_col, sort=False):
            idx = grp.index.to_numpy()
            if len(idx) < 2:
                continue
            for t in range(1, len(idx)):
                seq = feats[idx[max(0, t - lookback):t], :]
                x_one = torch.from_numpy(_pad_window(seq)).unsqueeze(0)
                context_pred.loc[idx[t]] = float(model(x_one).squeeze().cpu().item())

    context_cap = float(os.getenv("MAYA_GRU_CONTEXT_CAP", "0.32"))
    raw_adjust = (context_pred - base).clip(-context_cap, context_cap)
    conf = pd.to_numeric(out.get("sentiment_confidence"), errors="coerce").fillna(0.5).clip(0.0, 1.0)
    weight = (0.25 + 0.45 * (1.0 - conf)).clip(0.25, 0.70)
    adj = (raw_adjust * weight).clip(-context_cap, context_cap)

    updated_score = (base + adj).clip(-1.0, 1.0)
    weak_neutral = updated_score.abs().lt(0.06) & context_pred.abs().gt(0.12)
    updated_score = np.where(weak_neutral, 0.40 * updated_score + 0.60 * context_pred, updated_score)
    updated_score = pd.Series(updated_score, index=out.index, dtype="float64").clip(-1.0, 1.0)

    # If distribution is too compressed, expand dynamic range so sentiment does not collapse into neutral.
    abs_q90 = float(updated_score.abs().quantile(0.90)) if len(updated_score) else 0.0
    if abs_q90 < 0.07:
        scale = min(4.0, 0.14 / max(abs_q90, 1e-3))
        updated_score = (updated_score * scale).clip(-1.0, 1.0)
    out["score_gru_context"] = pd.to_numeric(context_pred, errors="coerce").fillna(0.0).clip(-1.0, 1.0)
    out["score_gru_adjustment"] = pd.to_numeric(adj, errors="coerce").fillna(0.0).clip(-1.0, 1.0)
    out[score_col] = updated_score.astype(float)
    out[label_col] = out[score_col].apply(polarity_label)

    if "score_context_adjustment" in out.columns:
        out["score_context_adjustment"] = (
            pd.to_numeric(out["score_context_adjustment"], errors="coerce").fillna(0.0) + out["score_gru_adjustment"]
        ).clip(-1.0, 1.0)

    if "sentiment_debug_flags" in out.columns:
        gru_flag = out["score_gru_adjustment"].abs().ge(0.02)
        out["sentiment_debug_flags"] = np.where(
            gru_flag,
            out["sentiment_debug_flags"].fillna("none").astype(str).apply(lambda s: s if s == "none" else f"{s}, gru"),
            out["sentiment_debug_flags"],
        )

    return out.sort_index()


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
def load_hf_pipelines(include_irony: bool = False):
    try:
        from transformers import pipeline

        sent_pipe = pipeline("sentiment-analysis", model=HF_SENTIMENT_MODEL, tokenizer=HF_SENTIMENT_MODEL, device=-1)
        irony_pipe = None
        if include_irony:
            irony_pipe = pipeline("text-classification", model=HF_IRONY_MODEL, tokenizer=HF_IRONY_MODEL, device=-1)
        return sent_pipe, irony_pipe
    except Exception:
        return None, None


def contextual_hf_sentiment(data: pd.DataFrame, context_window: int = 3) -> pd.DataFrame:
    sent_pipe, irony_pipe = load_hf_pipelines(include_irony=True)
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


def cardiff_sentiment_scores(
    data: pd.DataFrame,
    text_col: str = "message",
    group_col: str = "user_id",
    time_col: str = "created_at",
    context_window: int = 3,
) -> pd.DataFrame:
    if data.empty or text_col not in data.columns:
        return data

    # Prevent Streamlit UI freeze: if the user passes unfiltered raw data, bound it
    # We shouldn't execute raw HF inference on 6,000+ rows synchronously on page loads
    MAX_ROWS_HF = 500
    if len(data) > MAX_ROWS_HF:
        import sys
        print(f"[warning] Bypassing real-time HuggingFace sentiment inference (rows={len(data)} > limit={MAX_ROWS_HF}) to avoid UI hang.", file=sys.stderr)
        return data

    sent_pipe, _ = load_hf_pipelines(include_irony=False)
    if sent_pipe is None:
        return data

    out = data.copy()
    if group_col in out.columns and time_col in out.columns:
        out[time_col] = pd.to_datetime(out.get(time_col), errors="coerce", utc=True)
        out = out.sort_values([group_col, time_col], kind="mergesort").copy()

    use_context = os.getenv("MAYA_CARDIFF_USE_CONTEXT", "1").strip().lower() in {"1", "true", "yes"}
    inputs: list[str] = []
    if use_context and group_col in out.columns:
        for _, grp in out.groupby(group_col, sort=False):
            history: list[str] = []
            for _, row in grp.iterrows():
                msg = str(row.get(text_col, "")).strip()
                context = " ".join(history[-context_window:]).strip()
                if context:
                    inputs.append(f"Context: {context} [SEP] Current: {msg}")
                else:
                    inputs.append(msg)
                if msg:
                    history.append(msg[:240])
    else:
        inputs = out[text_col].fillna("").astype(str).tolist()

    if not inputs:
        return out

    preds = sent_pipe(inputs, truncation=True, max_length=256, batch_size=24)
    polarities: list[float] = []
    labels: list[str] = []
    confs: list[float] = []
    for rec in preds:
        lbl = _normalize_model_label(rec.get("label", "neutral"))
        score = float(rec.get("score", 0.0))
        if lbl == "positive":
            pol = score
        elif lbl == "negative":
            pol = -score
        else:
            pol = 0.0
        polarities.append(float(np.clip(pol, -1.0, 1.0)))
        labels.append(lbl)
        confs.append(float(np.clip(score, 0.0, 1.0)))

    out["sentiment_score"] = pd.Series(polarities, index=out.index, dtype="float64")
    out["sentiment_label"] = pd.Series(labels, index=out.index, dtype="object")
    out["sentiment_confidence"] = pd.Series(confs, index=out.index, dtype="float64")
    out["sentiment_source"] = "cardiffnlp_twitter_roberta"
    return out


def enforce_cardiff_sentiment(
    data: pd.DataFrame,
    text_col: str = "message",
    group_col: str = "user_id",
    time_col: str = "created_at",
    context_window: int = 4,
) -> pd.DataFrame:
    if data.empty:
        return data

    out = data.copy()
    scored = cardiff_sentiment_scores(
        out,
        text_col=text_col,
        group_col=group_col,
        time_col=time_col,
        context_window=context_window,
    )
    if not scored.empty and "sentiment_source" in scored.columns and scored["sentiment_source"].eq("cardiffnlp_twitter_roberta").any():
        out = scored

    out["sentiment_score"] = pd.to_numeric(out.get("sentiment_score"), errors="coerce").fillna(0.0).clip(-1.0, 1.0)
    if "sentiment_label" not in out.columns:
        out["sentiment_label"] = out["sentiment_score"].apply(polarity_label)
    else:
        lbl = out["sentiment_label"].fillna("").astype(str).str.lower().str.strip()
        valid = {"positive", "negative", "neutral"}
        out["sentiment_label"] = lbl.where(lbl.isin(valid), out["sentiment_score"].apply(polarity_label))

    conf_raw = out["sentiment_confidence"] if "sentiment_confidence" in out.columns else pd.Series(np.nan, index=out.index)
    conf = pd.to_numeric(conf_raw, errors="coerce")
    if not isinstance(conf, pd.Series):
        conf = pd.Series(conf, index=out.index)
    out["sentiment_confidence"] = conf.fillna(out["sentiment_score"].abs()).clip(0.0, 1.0)

    if "sentiment_source" not in out.columns:
        out["sentiment_source"] = "cardiff_unavailable_fallback"
    out["sentiment_source"] = out["sentiment_source"].fillna("cardiff_unavailable_fallback").astype(str)
    return out


@st.cache_data(show_spinner=False)
def load_sentiment_table(refresh_nonce: str | None = None) -> pd.DataFrame:
    _ = refresh_nonce
    # Fast path: if precomputed sentiment artifact exists, use it to avoid expensive re-inference on page load.
    if SENTIMENT_SCORES_PATH.exists():
        try:
            cached = pd.read_csv(SENTIMENT_SCORES_PATH)
            if not cached.empty:
                if "sentiment_score" in cached.columns:
                    cached = repair_flat_sentiment_scores(
                        cached,
                        text_col="message",
                        score_col="sentiment_score",
                        label_col="sentiment_label",
                    )
                needs_user_backfill = ("user_id" not in cached.columns) or cached["user_id"].isna().all()
                if needs_user_backfill:
                    if "session_id" in cached.columns and SESSIONS_SOURCE_PATH.exists():
                        sess_df = pd.read_csv(SESSIONS_SOURCE_PATH, usecols=["id", "user_id"])
                        sess_df["id"] = pd.to_numeric(sess_df["id"], errors="coerce")
                        sess_df["user_id"] = pd.to_numeric(sess_df["user_id"], errors="coerce")
                        cached["session_id"] = pd.to_numeric(cached["session_id"], errors="coerce")
                        cached = cached.merge(
                            sess_df.rename(columns={"id": "session_id", "user_id": "user_id_from_session"}),
                            on="session_id",
                            how="left",
                        )
                        if "user_id" in cached.columns:
                            cached["user_id"] = pd.to_numeric(cached["user_id"], errors="coerce")
                            cached["user_id"] = cached["user_id"].fillna(cached.get("user_id_from_session"))
                        else:
                            cached["user_id"] = pd.to_numeric(cached.get("user_id_from_session"), errors="coerce")
                        cached = cached.drop(columns=["user_id_from_session"], errors="ignore")

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

                # Trust the pipeline's scores and labels
                cached["polarity"] = pd.to_numeric(cached.get("sentiment_score"), errors="coerce").fillna(0.0)
                if "subjectivity" not in cached.columns:
                    cached["subjectivity"] = cached["polarity"].abs().clip(0.0, 1.0)
                cached["subjectivity"] = pd.to_numeric(cached.get("subjectivity"), errors="coerce").fillna(0.0)
                
                # Trust the pipeline's sentiment label completely. Only fallback to polarity_label if missing.
                lbl = cached.get("sentiment_label", "").fillna("").astype(str).str.lower().str.strip()
                cached["sentiment"] = lbl.where(lbl.isin(["positive", "negative", "neutral"]), cached["polarity"].apply(polarity_label))
                if "source" not in cached.columns:
                    cached["source"] = "user_message"
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
    if "sentiment_score" in data.columns:
        data = repair_flat_sentiment_scores(
            data,
            text_col="message",
            score_col="sentiment_score",
            label_col="sentiment_label",
        )
    data["user_id"] = pd.to_numeric(data["user_id"], errors="coerce")
    data = data.dropna(subset=["user_id"])
    data["user_id"] = data["user_id"].astype(int)
    data["message"] = data["message"].fillna("").astype(str).str.strip()
    data = data[data["message"].str.len() > 0]
    data["created_at"] = pd.to_datetime(data["created_at"], errors="coerce", utc=True)

    data = enforce_cardiff_sentiment(
        data,
        text_col="message",
        group_col="user_id",
        time_col="created_at",
        context_window=4,
    )
    data["polarity"] = pd.to_numeric(data.get("sentiment_score"), errors="coerce").fillna(0.0)
    data["subjectivity"] = data["polarity"].abs().clip(0.0, 1.0)
    lbl = data.get("sentiment_label", "").fillna("").astype(str).str.lower().str.strip()
    data["sentiment"] = lbl.where(lbl.isin(["positive", "negative", "neutral"]), data["polarity"].apply(polarity_label))
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


def build_tool_usage_signals(sentiment_df: pd.DataFrame) -> pd.DataFrame:
    # Pre-populate all categories and actions so they always appear in the dashboard
    rows: list[dict[str, object]] = []
    
    if not sentiment_df.empty:
        for _, row in sentiment_df.iterrows():
            msg_raw = str(row.get("message", ""))
            msg = msg_raw.lower()
            if not msg:
                continue
                
            pol = float(row.get("sentiment_score", row.get("polarity", 0.0)))
            # Align with lib/sentiment_utils.py threshold
            is_neg = pol < -0.15
            
            for category, actions in HUMANOID_TOOL_PATTERNS.items():
                for action_name, patterns in actions.items():
                    if any(p.search(msg) for p in patterns):
                        rows.append({
                            "category": category,
                            "tool_action": action_name,
                            "polarity": pol,
                            "is_negative": is_neg,
                            "sample_text": msg_raw
                        })

    req = pd.DataFrame(rows)
    
    # We want to ensure EVERY tool from the configuration is present, even with 0 mentions
    all_tools = []
    for category, actions in HUMANOID_TOOL_PATTERNS.items():
        for action_name in actions.keys():
            all_tools.append({"category": category, "tool_action": action_name})
    base_df = pd.DataFrame(all_tools)
    
    if req.empty:
        # No matches found, so everything is 0
        base_df["mentions"] = 0
        base_df["avg_polarity"] = 0.0
        base_df["neg_ratio"] = 0.0
        base_df["positive_sample"] = ""
        base_df["negative_sample"] = ""
        return base_df.sort_values(["category", "tool_action"])
        
    def get_pos_sample(group):
        if group.empty: return ""
        return group.loc[group["polarity"].idxmax()]["sample_text"]

    def get_neg_sample(group):
        if group.empty: return ""
        return group.loc[group["polarity"].idxmin()]["sample_text"]

    out = req.groupby(["category", "tool_action"], as_index=False).agg(
        mentions=("tool_action", "size"),
        avg_polarity=("polarity", "mean"),
        neg_ratio=("is_negative", "mean"),
    )
    
    # Manually add samples because idxmax/idxmin can't be easily put in agg() directly with other things
    pos_samples = req.groupby(["category", "tool_action"]).apply(get_pos_sample).reset_index(name="positive_sample")
    neg_samples = req.groupby(["category", "tool_action"]).apply(get_neg_sample).reset_index(name="negative_sample")
    
    out = out.merge(pos_samples, on=["category", "tool_action"], how="left")
    out = out.merge(neg_samples, on=["category", "tool_action"], how="left")
    
    # Merge with base_df to ensure 0-mention tools are included
    out = base_df.merge(out, on=["category", "tool_action"], how="left")
    out["mentions"] = out["mentions"].fillna(0).astype(int)
    out["avg_polarity"] = out["avg_polarity"].fillna(0.0)
    out["neg_ratio"] = out["neg_ratio"].fillna(0.0)
    out["positive_sample"] = out["positive_sample"].fillna("")
    out["negative_sample"] = out["negative_sample"].fillna("")
    
    return out.sort_values(["category", "mentions"], ascending=[True, False])


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
    sentiment_df = load_sentiment_table(str(refresh_nonce))
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
        "train_neg",
        "train_pos",
        "scale_pos_weight",
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

    for c in [
        "human_label_rows",
        "pseudo_label_rows",
        "joined_users",
        "train_rows",
        "test_rows",
        "train_neg",
        "train_pos",
    ]:
        if c in rep.columns:
            rep[c] = pd.to_numeric(rep[c], errors="coerce").fillna(0).astype(int)
    for c in ["accuracy", "auc", "scale_pos_weight"]:
        if c in rep.columns:
            rep[c] = pd.to_numeric(rep[c], errors="coerce")
    return rep


def load_xgb_user_predictions() -> pd.DataFrame:
    expected = ["user_id", "target", "pred_label", "pred_prob_positive", "predicted_class", "confidence", "confidence_gate"]
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
    # Derive confidence_gate if not present (backward compat with older prediction CSVs)
    if "confidence_gate" not in df.columns:
        conf = pd.to_numeric(df.get("confidence"), errors="coerce").fillna(0.0)
        df["confidence_gate"] = np.where(conf >= 0.70, "auto", np.where(conf >= 0.30, "review", "manual_review"))
    return df.sort_values("pred_prob_positive", ascending=False)


def get_xgb_model_artifact_status() -> tuple[bool, str]:
    for p in XGB_MODEL_PATH_CANDIDATES:
        if p.exists():
            mtime = pd.to_datetime(p.stat().st_mtime, unit="s").strftime("%Y-%m-%d %H:%M:%S")
            return True, f"{p.name} - updated {mtime}"
    return False, "Not found"


def compute_xgb_prediction_health(pred_df: pd.DataFrame) -> Dict[str, float | bool]:
    if pred_df.empty:
        return {
            "total": 0,
            "positive_rate": float("nan"),
            "negative_rate": float("nan"),
            "dominance": float("nan"),
            "probability_std": float("nan"),
            "avg_confidence": float("nan"),
            "confidence_std": float("nan"),
            "collapse_flag": False,
        }

    probs = pd.to_numeric(pred_df.get("pred_prob_positive"), errors="coerce").fillna(0.0)
    classes = pred_df.get("predicted_class", "").fillna("").astype(str).str.lower().str.strip()
    missing = classes.eq("")
    if missing.any():
        classes.loc[missing] = np.where(probs.loc[missing] >= 0.5, "positive", "negative")

    total = int(len(pred_df))
    pos_rate = float((classes == "positive").mean())
    neg_rate = float((classes == "negative").mean())
    dominance = float(max(pos_rate, neg_rate))
    prob_std = float(probs.std(ddof=0))
    confidence = pd.to_numeric(pred_df.get("confidence"), errors="coerce")
    if confidence.isna().all():
        confidence = (2.0 * (probs - 0.5).abs()).clip(0.0, 1.0)
    else:
        confidence = confidence.fillna((2.0 * (probs - 0.5).abs()).clip(0.0, 1.0))

    avg_conf = float(confidence.mean())
    conf_std = float(confidence.std(ddof=0))
    # Match training collapse heuristic: high dominance + very low probability spread.
    collapse_flag = bool(dominance >= 0.90 and prob_std <= 0.08)
    return {
        "total": total,
        "positive_rate": pos_rate,
        "negative_rate": neg_rate,
        "dominance": dominance,
        "probability_std": prob_std,
        "avg_confidence": avg_conf,
        "confidence_std": conf_std,
        "collapse_flag": collapse_flag,
    }


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
    if "persona_label" in table.columns:
        table["persona_label"] = table["persona_label"].astype(str).apply(simplify_persona_label)
    if "users" in profiles.columns:
        profiles["users"] = pd.to_numeric(profiles["users"], errors="coerce").fillna(0).astype(int)
    if "persona_label" in profiles.columns:
        profiles["persona_label"] = profiles["persona_label"].astype(str).apply(simplify_persona_label)
    if "importance" in importance.columns:
        importance["importance"] = pd.to_numeric(importance["importance"], errors="coerce").fillna(0.0)

    return table, profiles, importance


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
        if "persona_label" in df.columns:
            df["persona_label"] = df["persona_label"].astype(str).apply(simplify_persona_label)
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
    if "persona_label" in df.columns:
        df["persona_label"] = df["persona_label"].astype(str).apply(simplify_persona_label)
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

    needs_user_backfill = ("user_id" not in s.columns) or s["user_id"].isna().all()
    if needs_user_backfill:
        if "session_id" in s.columns and SESSIONS_SOURCE_PATH.exists():
            sess = pd.read_csv(SESSIONS_SOURCE_PATH, usecols=["id", "user_id"])
            sess["id"] = pd.to_numeric(sess["id"], errors="coerce")
            sess["user_id"] = pd.to_numeric(sess["user_id"], errors="coerce")
            s["session_id"] = pd.to_numeric(s["session_id"], errors="coerce")
            s = s.merge(
                sess.rename(columns={"id": "session_id", "user_id": "user_id_from_session"}),
                on="session_id",
                how="left",
            )
            if "user_id" in s.columns:
                s["user_id"] = pd.to_numeric(s["user_id"], errors="coerce")
                s["user_id"] = s["user_id"].fillna(s.get("user_id_from_session"))
            else:
                s["user_id"] = pd.to_numeric(s.get("user_id_from_session"), errors="coerce")
            s = s.drop(columns=["user_id_from_session"], errors="ignore")
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

    # Further lower thresholds to rebalance the distribution.
    # Using 85th and 65th percentiles with minimal floors to ensure
    # more users are flagged for review.
    q85 = float(agg["dissatisfaction_score"].quantile(0.85)) if not agg.empty else 0.0
    q65 = float(agg["dissatisfaction_score"].quantile(0.65)) if not agg.empty else 0.0
    high_cutoff = max(q85, 0.06)
    medium_cutoff = max(q65, 0.02)

    def bucket(v: float) -> str:
        if v >= high_cutoff:
            return "High"
        if v >= medium_cutoff:
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
def load_whatsapp_sentiment_messages(refresh_nonce: str | None = None) -> pd.DataFrame:
    _ = refresh_nonce
    cols = [
        "user_id",
        "message",
        "created_at",
        "sentiment_score",
        "sentiment_label",
        "sentiment_source",
        "sentiment_confidence",
        "score_model_raw",
        "heuristic_score",
        "score_rule_hint",
        "score_context_adjustment",
        "score_gru_context",
        "score_gru_adjustment",
        "sentiment_threshold_used",
        "sentiment_debug_flags",
        "role",
    ]
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

    needs_user_backfill = ("user_id" not in s.columns) or s["user_id"].isna().any()
    if needs_user_backfill:
        if "session_id" in s.columns and SESSIONS_SOURCE_PATH.exists():
            sess = pd.read_csv(SESSIONS_SOURCE_PATH, usecols=["id", "user_id"])
            sess["id"] = pd.to_numeric(sess["id"], errors="coerce")
            sess["user_id_from_session"] = pd.to_numeric(sess["user_id"], errors="coerce")
            s["session_id"] = pd.to_numeric(s.get("session_id"), errors="coerce")
            s = s.merge(
                sess[["id", "user_id_from_session"]].rename(columns={"id": "session_id"}),
                on="session_id",
                how="left",
            )
            if "user_id" in s.columns:
                s["user_id"] = s["user_id"].fillna(s["user_id_from_session"])
            else:
                s["user_id"] = s["user_id_from_session"]
            s = s.drop(columns=["user_id_from_session"], errors="ignore")
        elif "user_id" not in s.columns:
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

    s = enforce_cardiff_sentiment(
        s,
        text_col="message",
        group_col="user_id",
        time_col="created_at",
        context_window=4,
    )

    if "role" not in s.columns:
        s["role"] = "user"
    for c in cols:
        if c not in s.columns:
            s[c] = np.nan
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

        steps = build_steps(include_redis_publish=False, include_kafka_publish=True)
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
            {"step_id": "publish_raw_csv_to_kafka", "description": "Publish raw CSV data to Kafka topics", "command": f"{py} -m pipelines.ingestion.kafka_csv_producer --broker localhost:9092 --delay 0.0"},
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

