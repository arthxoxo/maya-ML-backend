"""
Feature Engineering Pipeline — Per-User Behavioral Feature Matrix

Reads all 4 CSVs and builds ~50 engineered features per user across:
  A. Message behavior (from whatsapp_messages)
  B. Temporal patterns
  C. Sentiment trajectory
  D. Session engagement
  E. Feedback signals

Output: user_feature_matrix.csv — one row per user, ready for ML.

Usage:
    source flink_venv/bin/activate
    python feature_engineering.py
"""

import os
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime
import warnings
import re
from tqdm import tqdm

from app_config import FEATURE_OUTPUT_DIR, RAW_DATA_DIR, SENTIMENT_ARTIFACT_DIR

warnings.filterwarnings("ignore")

# ── Configuration ────────────────────────────────────────────────────────────

DATA_DIR = RAW_DATA_DIR
OUTPUT_DIR = FEATURE_OUTPUT_DIR
HF_SENTIMENT_MODEL = "cardiffnlp/twitter-roberta-base-sentiment-latest"
_HF_PIPE = None
_HF_UNAVAILABLE = False
_NEGATIVE_TERMS = {
    "bad", "worse", "worst", "hate", "angry", "upset", "frustrated", "annoyed",
    "terrible", "awful", "slow", "broken", "error", "issue", "problem", "failed",
}
_POSITIVE_TERMS = {
    "good", "great", "awesome", "nice", "love", "happy", "thanks", "thankyou",
    "resolved", "perfect", "excellent", "fast", "smooth",
}

# ── Data Loading ─────────────────────────────────────────────────────────────


def _find_csv(folder: Path, name: str) -> Path:
    """Try finding 'maya_name.csv' then 'name.csv'."""
    p1 = folder / f"maya_{name}"
    p2 = folder / name
    if p1.exists():
        return p1
    if p2.exists():
        return p2
    # Fallback to direct name so pd.read_csv fails with a clear message if missing
    return p2


def load_data():
    """Load all CSVs with proper date parsing."""
    print("📂  Loading CSVs...")

    users = pd.read_csv(_find_csv(DATA_DIR, "users.csv"))
    users["created_at"] = pd.to_datetime(users["created_at"], format="mixed", utc=True, errors="coerce")

    messages = pd.read_csv(_find_csv(DATA_DIR, "whatsapp_messages.csv"))
    messages["created_at"] = pd.to_datetime(messages["created_at"], format="mixed", utc=True, errors="coerce")

    sessions = pd.read_csv(_find_csv(DATA_DIR, "sessions.csv"))
    sessions["created_at"] = pd.to_datetime(sessions["created_at"], format="mixed", utc=True, errors="coerce")

    feedbacks = pd.read_csv(_find_csv(DATA_DIR, "feedbacks.csv"))
    feedbacks["created_at"] = pd.to_datetime(feedbacks["created_at"], format="mixed", utc=True, errors="coerce")

    print(f"    Users:    {len(users):>7,}")
    print(f"    Messages: {len(messages):>7,}")
    print(f"    Sessions: {len(sessions):>7,}")
    print(f"    Feedback: {len(feedbacks):>7,}")

    return users, messages, sessions, feedbacks


# ── Helper: Sentiment ────────────────────────────────────────────────────────


def _get_hf_pipeline():
    global _HF_PIPE, _HF_UNAVAILABLE
    if _HF_PIPE is not None:
        return _HF_PIPE
    if _HF_UNAVAILABLE:
        return None
    try:
        import sys
        from pathlib import Path as _Path
        sys.path.insert(0, str(_Path(__file__).resolve().parents[2]))
        from lib.device_utils import resolve_device
        from transformers import pipeline

        _dev = resolve_device()
        _HF_PIPE = pipeline(
            "sentiment-analysis",
            model=HF_SENTIMENT_MODEL,
            tokenizer=HF_SENTIMENT_MODEL,
            device=str(_dev),
        )
    except Exception:
        _HF_UNAVAILABLE = True
        _HF_PIPE = None
    return _HF_PIPE


def _heuristic_sentiment_subjectivity(text: str) -> tuple[float, float]:
    s = str(text or "").strip().lower()
    if not s:
        return 0.0, 0.0

    tokens = re.findall(r"[a-z']+", s)
    if not tokens:
        return 0.0, 0.0

    pos_hits = sum(1 for t in tokens if t in _POSITIVE_TERMS)
    neg_hits = sum(1 for t in tokens if t in _NEGATIVE_TERMS)
    polarity = (pos_hits - neg_hits) / max(len(tokens), 6)
    polarity = float(max(min(polarity * 2.0, 1.0), -1.0))
    subjectivity = float(min(max(0.2 + 0.8 * abs(polarity), 0.0), 1.0))
    return round(polarity, 4), round(subjectivity, 4)


def score_texts_sentiment(texts: list[str], batch_size: int = 32) -> tuple[np.ndarray, np.ndarray]:
    if not texts:
        return np.array([], dtype=np.float32), np.array([], dtype=np.float32)

    fast_mode = os.getenv("MAYA_PIPELINE_FAST", "0").lower() in ("1", "true", "yes")
    pipe = _get_hf_pipeline() if not fast_mode else None

    if pipe is None:
        if fast_mode:
            print(f"    [Fast Mode] Using heuristic sentiment for {len(texts):,} messages...")
        vals = [_heuristic_sentiment_subjectivity(t) for t in texts]
        pol = np.array([v[0] for v in vals], dtype=np.float32)
        subj = np.array([v[1] for v in vals], dtype=np.float32)
        return pol, subj

    print(f"    [Transformer] Scoring {len(texts):,} messages via {HF_SENTIMENT_MODEL}...")
    try:
        results_pol = []
        results_subj = []
        # Process in chunks with tqdm progress bar
        for i in tqdm(range(0, len(texts), batch_size), desc="    Sentiment", unit="batch", leave=False):
            chunk = texts[i : i + batch_size]
            out = pipe(chunk, truncation=True, max_length=256)
            for rec in out:
                label = str(rec.get("label", "")).strip().lower()
                score = float(rec.get("score", 0.0))
                if "positive" in label or label in {"label_2", "2"}:
                    p = score
                elif "negative" in label or label in {"label_0", "0"}:
                    p = -score
                else:
                    p = 0.0
                s = min(max(0.3 + 0.7 * abs(p), 0.0), 1.0)
                results_pol.append(round(float(max(min(p, 1.0), -1.0)), 4))
                results_subj.append(round(float(s), 4))

        return np.array(results_pol, dtype=np.float32), np.array(results_subj, dtype=np.float32)
    except Exception as exc:
        print(f"    [Check] Pipeline error, falling back to heuristic: {exc}")
        vals = [_heuristic_sentiment_subjectivity(t) for t in texts]
        pol = np.array([v[0] for v in vals], dtype=np.float32)
        subj = np.array([v[1] for v in vals], dtype=np.float32)
        return pol, subj


def load_precomputed_sentiment(user_msgs: pd.DataFrame) -> pd.DataFrame:
    """
    Attach precomputed sentiment from artifacts/sentiment/sentiment_scores.csv.
    Falls back to empty sentiment columns when artifact is missing/unusable.
    """
    enriched = user_msgs.copy()
    artifact_path = SENTIMENT_ARTIFACT_DIR / "sentiment_scores.csv"

    if not artifact_path.exists():
        print(f"    [Sentiment] Precomputed artifact not found at {artifact_path}")
        enriched["sentiment"] = np.nan
        enriched["subjectivity"] = np.nan
        return enriched

    try:
        scores = pd.read_csv(artifact_path)
    except Exception as exc:
        print(f"    [Sentiment] Failed to read precomputed artifact ({exc}).")
        enriched["sentiment"] = np.nan
        enriched["subjectivity"] = np.nan
        return enriched

    if "sentiment_score" not in scores.columns:
        print("    [Sentiment] sentiment_scores.csv missing 'sentiment_score' column.")
        enriched["sentiment"] = np.nan
        enriched["subjectivity"] = np.nan
        return enriched

    score_cols = [c for c in ["id", "sentiment_score", "sentiment_confidence"] if c in scores.columns]
    if "id" in score_cols and "id" in enriched.columns:
        merged = enriched.merge(scores[score_cols], on="id", how="left")
    else:
        # Fallback key when message id is unavailable in either dataset.
        fallback_keys = [c for c in ["session_id", "created_at", "message"] if c in scores.columns and c in enriched.columns]
        if len(fallback_keys) < 2:
            print("    [Sentiment] Could not align precomputed sentiment to messages; missing join keys.")
            enriched["sentiment"] = np.nan
            enriched["subjectivity"] = np.nan
            return enriched
        merged = enriched.merge(scores[fallback_keys + [c for c in ["sentiment_score", "sentiment_confidence"] if c in scores.columns]], on=fallback_keys, how="left")

    merged["sentiment"] = pd.to_numeric(merged.get("sentiment_score"), errors="coerce")
    conf = pd.to_numeric(merged.get("sentiment_confidence"), errors="coerce")
    merged["subjectivity"] = conf.where(conf.notna(), merged["sentiment"].abs()).clip(0.0, 1.0)

    matched = int(merged["sentiment"].notna().sum())
    total = int(len(merged))
    print(f"    [Sentiment] Loaded precomputed scores for {matched:,}/{total:,} user messages.")
    return merged.drop(columns=["sentiment_score", "sentiment_confidence"], errors="ignore")


# ── Map messages to users via sessions ────────────────────────────────────────


def map_messages_to_users(messages, sessions):
    """
    WhatsApp messages link to sessions via session_id.
    Sessions link to users via user_id.
    """
    session_user_map = sessions[["id", "user_id"]].drop_duplicates()
    session_user_map = session_user_map.rename(columns={"id": "session_id"})

    merged = messages.merge(session_user_map, on="session_id", how="left")
    print(f"    Messages mapped to users: {merged['user_id'].notna().sum():,} / {len(merged):,}")
    return merged


# ══════════════════════════════════════════════════════════════════════════════
# FEATURE GROUP A: Message Behavior Features
# ══════════════════════════════════════════════════════════════════════════════


def build_message_features(messages):
    """Per-user features from WhatsApp message data."""
    print("\n🔧  Building message behavior features...")

    user_msgs = messages[messages["role"] == "user"].copy()
    assistant_msgs = messages[messages["role"] == "assistant"].copy()

    # ── Basic counts ─────────────────────────────────────────────────────
    user_counts = (
        user_msgs.groupby("user_id")
        .agg(
            total_messages_sent=("id", "count"),
            unique_sessions=("session_id", "nunique"),
        )
        .reset_index()
    )

    # ── Message length stats ─────────────────────────────────────────────
    user_msgs["msg_len"] = user_msgs["message"].fillna("").str.len()

    length_stats = (
        user_msgs.groupby("user_id")["msg_len"]
        .agg(["mean", "std", "max"])
        .rename(columns={"mean": "avg_message_length", "std": "msg_length_std", "max": "max_message_length"})
        .reset_index()
    )

    # ── Question ratio ───────────────────────────────────────────────────
    question_words = r"^(what|how|why|when|where|who|which|can|could|would|is|are|do|does|did|will|shall)\b"
    user_msgs["is_question"] = (
        user_msgs["message"].fillna("").str.strip().str.endswith("?")
        | user_msgs["message"].fillna("").str.lower().str.match(question_words)
    )
    question_ratio = (
        user_msgs.groupby("user_id")["is_question"]
        .mean()
        .rename("question_ratio")
        .reset_index()
    )

    # ── Conversation depth (messages per session) ────────────────────────
    conv_depth = (
        user_msgs.groupby("user_id")
        .apply(lambda x: x.groupby("session_id").size().mean(), include_groups=False)
        .rename("avg_conversation_depth")
        .reset_index()
    )

    # ── Tool usage rate (sessions where Maya used tools) ─────────────────
    tool_msgs = messages[messages["tool_calls"].notna()]
    tool_sessions = tool_msgs["session_id"].unique()
    user_sessions = user_msgs.groupby("user_id")["session_id"].apply(set).reset_index()
    user_sessions["tool_usage_rate"] = user_sessions["session_id"].apply(
        lambda s: len(s & set(tool_sessions)) / max(len(s), 1)
    )
    tool_rate = user_sessions[["user_id", "tool_usage_rate"]]

    # ── Token & cost stats (from assistant replies) ──────────────────────
    cost_stats = (
        assistant_msgs.groupby("user_id")
        .agg(
            total_input_tokens=("input_tokens", "sum"),
            total_output_tokens=("output_tokens", "sum"),
            total_cost_usd=("cost_usd", "sum"),
        )
        .reset_index()
    )

    # ── Response time (user msg → next assistant msg in same session) ─────
    msgs_sorted = messages.sort_values(["session_id", "created_at"])
    msgs_sorted["next_role"] = msgs_sorted.groupby("session_id")["role"].shift(-1)
    msgs_sorted["next_time"] = msgs_sorted.groupby("session_id")["created_at"].shift(-1)

    user_then_assistant = msgs_sorted[
        (msgs_sorted["role"] == "user") & (msgs_sorted["next_role"] == "assistant")
    ].copy()
    user_then_assistant["response_time_sec"] = (
        user_then_assistant["next_time"] - user_then_assistant["created_at"]
    ).dt.total_seconds()

    response_stats = (
        user_then_assistant.groupby("user_id")["response_time_sec"]
        .agg(["mean", "median"])
        .rename(columns={"mean": "avg_response_time_sec", "median": "median_response_time_sec"})
        .reset_index()
    )

    # ── Merge all message features ───────────────────────────────────────
    feat = user_counts
    for df in [length_stats, question_ratio, conv_depth, tool_rate, cost_stats, response_stats]:
        feat = feat.merge(df, on="user_id", how="left")

    print(f"    ✅ {len(feat.columns) - 1} message features built")
    return feat


# ══════════════════════════════════════════════════════════════════════════════
# FEATURE GROUP B: Temporal Pattern Features
# ══════════════════════════════════════════════════════════════════════════════


def build_temporal_features(messages):
    """Per-user temporal behavior patterns."""
    print("🔧  Building temporal pattern features...")

    user_msgs = messages[messages["role"] == "user"].copy()
    user_msgs["hour"] = user_msgs["created_at"].dt.hour
    user_msgs["day_of_week"] = user_msgs["created_at"].dt.dayofweek  # 0=Mon
    user_msgs["date"] = user_msgs["created_at"].dt.date

    feats = []
    for uid, grp in user_msgs.groupby("user_id"):
        dates = grp["date"].unique()
        daily_counts = grp.groupby("date").size()
        hours = grp["hour"]
        days = grp["day_of_week"]

        feat = {
            "user_id": uid,
            # Peak usage
            "peak_usage_hour": hours.mode().iloc[0] if len(hours) > 0 else None,
            "peak_usage_day": days.mode().iloc[0] if len(days) > 0 else None,
            # Time-of-day distribution
            "morning_ratio": ((hours >= 6) & (hours < 12)).mean(),    # 6AM-12PM
            "afternoon_ratio": ((hours >= 12) & (hours < 18)).mean(), # 12PM-6PM
            "evening_ratio": ((hours >= 18) & (hours < 22)).mean(),   # 6PM-10PM
            "night_ratio": ((hours >= 22) | (hours < 6)).mean(),      # 10PM-6AM
            # Weekend vs weekday
            "weekend_ratio": (days >= 5).mean(),
            # Activity consistency
            "active_days_count": len(dates),
            "messages_per_active_day": daily_counts.mean(),
            "activity_std": daily_counts.std() if len(daily_counts) > 1 else 0,
            # Account timeline
            "first_seen": grp["created_at"].min(),
            "last_seen": grp["created_at"].max(),
            "account_age_days": (grp["created_at"].max() - grp["created_at"].min()).days,
            # Gaps
            "longest_inactive_gap_days": (
                pd.Series(sorted(dates)).diff().max().days
                if len(dates) > 1 else 0
            ),
            "days_since_last_activity": (
                (pd.Timestamp.now(tz="UTC") - grp["created_at"].max()).days
            ),
        }
        feats.append(feat)

    result = pd.DataFrame(feats)

    # Message trend: compare last 30-day count vs first 30-day count
    for uid in result["user_id"]:
        grp = user_msgs[user_msgs["user_id"] == uid]
        mid_date = grp["created_at"].min() + (grp["created_at"].max() - grp["created_at"].min()) / 2
        first_half = len(grp[grp["created_at"] <= mid_date])
        second_half = len(grp[grp["created_at"] > mid_date])
        if first_half > 0:
            trend = (second_half - first_half) / first_half
        else:
            trend = 0.0
        result.loc[result["user_id"] == uid, "message_count_trend"] = round(trend, 3)

    # Drop datetime columns (keep only numeric features)
    result = result.drop(columns=["first_seen", "last_seen"], errors="ignore")

    print(f"    ✅ {len(result.columns) - 1} temporal features built")
    return result


# ══════════════════════════════════════════════════════════════════════════════
# FEATURE GROUP C: Sentiment Features
# ══════════════════════════════════════════════════════════════════════════════


def build_sentiment_features(messages):
    """Per-user sentiment trajectory from their messages."""
    print("🔧  Building sentiment features...")

    user_msgs = messages[messages["role"] == "user"].copy()

    print("    Loading precomputed sentiment artifact (RoBERTa output)...")
    user_msgs = load_precomputed_sentiment(user_msgs)

    missing_mask = user_msgs["sentiment"].isna()
    missing_count = int(missing_mask.sum())
    if missing_count:
        print(f"    Computing fallback sentiment for {missing_count:,} unmatched messages...")
        texts = user_msgs.loc[missing_mask, "message"].fillna("").astype(str).tolist()
        pol, subj = score_texts_sentiment(texts)
        user_msgs.loc[missing_mask, "sentiment"] = pol
        user_msgs.loc[missing_mask, "subjectivity"] = subj

    feats = []
    for uid, grp in user_msgs.groupby("user_id"):
        sents = grp["sentiment"]
        subjs = grp["subjectivity"]
        sorted_grp = grp.sort_values("created_at")

        feat = {
            "user_id": uid,
            # Overall sentiment
            "avg_sentiment": sents.mean(),
            "sentiment_std": sents.std() if len(sents) > 1 else 0,
            "min_sentiment": sents.min(),
            "max_sentiment": sents.max(),
            # Polarity distribution
            "negative_msg_ratio": (sents < -0.05).mean(),
            "neutral_msg_ratio": ((sents >= -0.05) & (sents <= 0.05)).mean(),
            "positive_msg_ratio": (sents > 0.05).mean(),
            # Subjectivity
            "avg_subjectivity": subjs.mean(),
            # Sentiment trend (linear slope over time)
            "sentiment_trend": np.nan,
            # Emotional volatility (sentiment swing between consecutive messages)
            "sentiment_volatility": sents.diff().abs().mean() if len(sents) > 1 else 0,
        }

        # Compute sentiment trend (slope of linear fit over message index)
        if len(sorted_grp) >= 3:
            x = np.arange(len(sorted_grp))
            y = sorted_grp["sentiment"].values
            try:
                slope = np.polyfit(x, y, 1)[0]
                feat["sentiment_trend"] = round(slope, 6)
            except (np.linalg.LinAlgError, ValueError):
                feat["sentiment_trend"] = 0.0

        feats.append(feat)

    result = pd.DataFrame(feats)
    print(f"    ✅ {len(result.columns) - 1} sentiment features built")
    return result


# ══════════════════════════════════════════════════════════════════════════════
# FEATURE GROUP D: Session Engagement Features
# ══════════════════════════════════════════════════════════════════════════════


def build_session_features(sessions):
    """Per-user session engagement features."""
    print("🔧  Building session engagement features...")

    feats = []
    for uid, grp in sessions.groupby("user_id"):
        sorted_grp = grp.sort_values("created_at")
        durations = grp["duration"]
        gaps = sorted_grp["created_at"].diff().dt.total_seconds() / 3600  # hours

        feat = {
            "user_id": uid,
            "total_sessions": len(grp),
            "avg_session_duration_sec": durations.mean(),
            "max_session_duration_sec": durations.max(),
            "total_session_duration_sec": durations.sum(),
            "sessions_with_duration": (durations > 0).sum(),
            "session_completion_rate": (durations > 0).mean(),
            # Session frequency
            "avg_session_gap_hours": gaps.mean() if len(gaps) > 1 else None,
            "min_session_gap_hours": gaps.min() if len(gaps) > 1 else None,
            # Content richness (non-null summaries, transcriptions)
            "has_transcription_rate": grp["transcription"].notna().mean(),
            "has_summary_rate": grp["summary"].notna().mean(),
            # Sessions per week
            "sessions_per_week": (
                len(grp)
                / max(
                    (grp["created_at"].max() - grp["created_at"].min()).days / 7,
                    1,
                )
            ),
            # Duration trend
            "session_duration_trend": np.nan,
        }

        # Duration trend (slope)
        active = grp[grp["duration"] > 0]
        if len(active) >= 3:
            x = np.arange(len(active))
            y = active["duration"].values.astype(float)
            try:
                feat["session_duration_trend"] = round(np.polyfit(x, y, 1)[0], 4)
            except (np.linalg.LinAlgError, ValueError):
                feat["session_duration_trend"] = 0.0

        feats.append(feat)

    result = pd.DataFrame(feats)
    print(f"    ✅ {len(result.columns) - 1} session features built")
    return result


# ══════════════════════════════════════════════════════════════════════════════
# FEATURE GROUP E: Feedback Features
# ══════════════════════════════════════════════════════════════════════════════


def build_feedback_features(feedbacks):
    """Per-user feedback-derived features."""
    print("🔧  Building feedback features...")

    feedback_texts = feedbacks["message"].fillna("").astype(str).tolist()
    feedback_polarity, _ = score_texts_sentiment(feedback_texts)
    feedbacks["sentiment"] = feedback_polarity

    feats = []
    for uid, grp in feedbacks.groupby("user_id"):
        feat = {
            "user_id": uid,
            "feedback_count": len(grp),
            "feedback_avg_sentiment": grp["sentiment"].mean(),
            "feedback_min_sentiment": grp["sentiment"].min(),
            "has_negative_feedback": int((grp["sentiment"] < -0.05).any()),
        }
        feats.append(feat)

    result = pd.DataFrame(feats)
    print(f"    ✅ {len(result.columns) - 1} feedback features built")
    return result


# ══════════════════════════════════════════════════════════════════════════════
# FEATURE GROUP F: NLP Complexity Features
# ══════════════════════════════════════════════════════════════════════════════


def build_nlp_features(messages):
    """Per-user NLP-derived text complexity features."""
    print("🔧  Building NLP complexity features...")

    user_msgs = messages[messages["role"] == "user"].copy()
    user_msgs["text"] = user_msgs["message"].fillna("")

    feats = []
    for uid, grp in user_msgs.groupby("user_id"):
        texts = grp["text"].tolist()
        all_text = " ".join(texts)
        words = all_text.split()
        unique_words = set(w.lower() for w in words)

        # Word counts per message
        word_counts = [len(t.split()) for t in texts]

        # Emoji detection (basic Unicode emoji range)
        emoji_pattern = re.compile(
            "[\U0001F600-\U0001F64F\U0001F300-\U0001F5FF\U0001F680-\U0001F6FF"
            "\U0001F1E0-\U0001F1FF\U00002702-\U000027B0\U0001F900-\U0001F9FF"
            "\U0001FA00-\U0001FA6F\U0001FA70-\U0001FAFF]+",
            flags=re.UNICODE,
        )
        emoji_count = sum(len(emoji_pattern.findall(t)) for t in texts)

        feat = {
            "user_id": uid,
            "total_words": len(words),
            "unique_words": len(unique_words),
            "vocabulary_richness": len(unique_words) / max(len(words), 1),
            "avg_words_per_message": np.mean(word_counts) if word_counts else 0,
            "emoji_usage_rate": emoji_count / max(len(texts), 1),
            "avg_sentence_length_chars": np.mean([len(t) for t in texts]) if texts else 0,
            # Short vs long messages
            "short_msg_ratio": sum(1 for wc in word_counts if wc <= 3) / max(len(word_counts), 1),
            "long_msg_ratio": sum(1 for wc in word_counts if wc >= 20) / max(len(word_counts), 1),
        }
        feats.append(feat)

    result = pd.DataFrame(feats)
    print(f"    ✅ {len(result.columns) - 1} NLP complexity features built")
    return result


# ══════════════════════════════════════════════════════════════════════════════
# MAIN: Assemble the Feature Matrix
# ══════════════════════════════════════════════════════════════════════════════


def main():
    print("=" * 60)
    print("  Maya ML — Feature Engineering Pipeline")
    print("=" * 60)

    # ── Load ─────────────────────────────────────────────────────────────
    users, messages, sessions, feedbacks = load_data()

    # ── Map messages to users ────────────────────────────────────────────
    messages = map_messages_to_users(messages, sessions)

    # ── Build feature groups ─────────────────────────────────────────────
    msg_features = build_message_features(messages)
    temporal_features = build_temporal_features(messages)
    sentiment_features = build_sentiment_features(messages)
    session_features = build_session_features(sessions)
    feedback_features = build_feedback_features(feedbacks)
    nlp_features = build_nlp_features(messages)

    # ── Merge into single user-level matrix ──────────────────────────────
    print("\n🔗  Merging all feature groups...")

    # Start with user identity
    feature_matrix = users[["id", "first_name", "last_name", "status", "type", "timezone"]].copy()
    feature_matrix = feature_matrix.rename(columns={"id": "user_id"})
    feature_matrix["user_name"] = (
        feature_matrix["first_name"].fillna("") + " " + feature_matrix["last_name"].fillna("")
    ).str.strip()

    # Merge all feature groups
    for df in [msg_features, temporal_features, sentiment_features,
               session_features, feedback_features, nlp_features]:
        feature_matrix = feature_matrix.merge(df, on="user_id", how="left")

    # ── Derived composite scores ─────────────────────────────────────────
    print("🔧  Computing composite scores...")

    # Engagement Score (normalized 0-1)
    if "total_messages_sent" in feature_matrix.columns:
        msgs_norm = feature_matrix["total_messages_sent"].rank(pct=True)
        sessions_norm = feature_matrix["total_sessions"].rank(pct=True)
        active_days_norm = feature_matrix["active_days_count"].rank(pct=True)
        feature_matrix["engagement_score"] = (
            (msgs_norm * 0.4 + sessions_norm * 0.3 + active_days_norm * 0.3)
        ).round(4)

    # Satisfaction Proxy (from sentiment + feedback)
    feature_matrix["satisfaction_proxy"] = feature_matrix["avg_sentiment"].fillna(0)
    if "feedback_avg_sentiment" in feature_matrix.columns:
        has_fb = feature_matrix["feedback_avg_sentiment"].notna()
        feature_matrix.loc[has_fb, "satisfaction_proxy"] = (
            feature_matrix.loc[has_fb, "avg_sentiment"].fillna(0) * 0.6
            + feature_matrix.loc[has_fb, "feedback_avg_sentiment"] * 0.4
        )

    # ── Fill NaN for users with no data ──────────────────────────────────
    numeric_cols = feature_matrix.select_dtypes(include=[np.number]).columns
    feature_matrix[numeric_cols] = feature_matrix[numeric_cols].fillna(0)

    # ── Save ─────────────────────────────────────────────────────────────
    output_path = OUTPUT_DIR / "user_feature_matrix.csv"
    feature_matrix.to_csv(output_path, index=False)

    print(f"\n{'═' * 60}")
    print(f"✅  Feature matrix saved: {output_path}")
    print(f"    Shape: {feature_matrix.shape[0]} users × {feature_matrix.shape[1]} features")
    print(f"{'═' * 60}")

    # ── Summary ──────────────────────────────────────────────────────────
    print("\n📊  Feature Groups Summary:")
    groups = {
        "Message Behavior": msg_features.columns.drop("user_id").tolist(),
        "Temporal Patterns": temporal_features.columns.drop("user_id").tolist(),
        "Sentiment": sentiment_features.columns.drop("user_id").tolist(),
        "Session Engagement": session_features.columns.drop("user_id").tolist(),
        "Feedback": feedback_features.columns.drop("user_id").tolist(),
        "NLP Complexity": nlp_features.columns.drop("user_id").tolist(),
        "Composite": ["engagement_score", "satisfaction_proxy"],
    }
    total = 0
    for group_name, cols in groups.items():
        print(f"    {group_name:25s}  {len(cols):>3} features")
        total += len(cols)
    print(f"    {'─' * 35}")
    print(f"    {'TOTAL':25s}  {total:>3} features")

    # Preview top engaged users
    print("\n🏆  Top 5 Users by Engagement Score:")
    top = feature_matrix.nlargest(5, "engagement_score")[
        ["user_name", "total_messages_sent", "total_sessions", "avg_sentiment", "engagement_score"]
    ]
    print(top.to_string(index=False))

    return feature_matrix


if __name__ == "__main__":
    main()
