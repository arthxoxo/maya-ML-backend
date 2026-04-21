"""
Build GNN node tables from Flink filesystem outputs.

Reads Flink-derived datasets from:
  flink_engineered/users
  flink_engineered/sessions
  flink_engineered/feedbacks
  flink_engineered/messages_sentiment

Also supports legacy fallback path:
  engineered_features/*

Writes:
  gnn_preprocessed/users_nodes.csv
  gnn_preprocessed/sessions_nodes.csv
  gnn_preprocessed/messages_nodes.csv
  gnn_preprocessed/feedback_nodes.csv
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from config import BASE_DIR, FLINK_ENGINEERED_DIR, GNN_PREPROCESSED_DIR, SECRET_DATA_DIR
from lib.online_store import save_artifact_df


OUT_DIR = GNN_PREPROCESSED_DIR


def resolve_flink_input_dir() -> Path:
    candidates = [
        FLINK_ENGINEERED_DIR,
        BASE_DIR / "engineered_features",  # legacy name used in older scripts
    ]
    for p in candidates:
        if (p / "messages_sentiment").exists():
            return p
    return FLINK_ENGINEERED_DIR


def read_flink_dir(dir_path: Path) -> pd.DataFrame:
    if not dir_path.exists():
        return pd.DataFrame()
    files = sorted(
        [
            p
            for p in dir_path.rglob("*")
            if p.is_file()
            # Flink streaming sinks write files named .part-*.inprogress
            # Accept both finalised files and in-progress part files.
            # Skip only Flink internal checkpoint/metadata files (_SUCCESS, .crc etc.)
            and not p.name.startswith("_")
            and not p.suffix in {".crc", ".json"}
            and "checkpoint" not in str(p)
        ]
    )
    if not files:
        return pd.DataFrame()

    frames = []
    for fp in files:
        try:
            frames.append(pd.read_csv(fp, header=None))
        except Exception:
            continue
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


def read_secret_csv(filename: str) -> pd.DataFrame:
    p = SECRET_DATA_DIR / filename
    if not p.exists():
        return pd.DataFrame()
    try:
        return pd.read_csv(p)
    except Exception:
        return pd.DataFrame()


def align_columns(df: pd.DataFrame, expected_cols: list[str]) -> pd.DataFrame:
    out = df.copy()
    for c in expected_cols:
        if c not in out.columns:
            out[c] = pd.NA
    return out[expected_cols]


def normalize_raw_table(df: pd.DataFrame, expected_cols: list[str]) -> pd.DataFrame:
    if df.empty:
        return df
    # Headerless sink files read as numbered columns.
    if all(isinstance(c, int) for c in df.columns):
        if df.shape[1] >= len(expected_cols):
            out = df.iloc[:, : len(expected_cols)].copy()
            out.columns = expected_cols
            return out
        out = df.copy()
        for _ in range(len(expected_cols) - out.shape[1]):
            out[out.shape[1]] = pd.NA
        out = out.iloc[:, : len(expected_cols)].copy()
        out.columns = expected_cols
        return out

    # Headered files: keep known columns and add missing.
    if set(expected_cols).issubset(set(df.columns)):
        return df[expected_cols].copy()
    return align_columns(df, expected_cols)


def main() -> None:
    flink_dir = resolve_flink_input_dir()
    print(f"Using Flink input directory: {flink_dir}")

    users_raw = read_flink_dir(flink_dir / "users")
    sessions_raw = read_flink_dir(flink_dir / "sessions")
    feedbacks_raw = read_flink_dir(flink_dir / "feedbacks")
    messages_raw = read_flink_dir(flink_dir / "messages_sentiment")

    users_cols = [
        "user_id", "created_at", "updated_at", "deleted_at", "first_name", "last_name",
        "timezone", "country", "status", "type", "longitude", "latitude", "contacts_backfilled",
    ]
    sessions_cols = [
        "session_id", "user_id", "created_at", "updated_at", "deleted_at", "duration",
        "billed_duration", "transcription", "summary", "provider",
    ]
    feedback_cols = [
        "feedback_id", "user_id", "session_id", "message", "feedback_source", "created_at",
        "updated_at", "deleted_at",
    ]
    messages_cols = [
        "message_id", "session_id", "sender_user_id", "role", "message", "created_at", "updated_at",
        "deleted_at", "input_tokens", "output_tokens", "model_name", "cost_usd", "recipient_name",
        "status", "sentiment_score", "sentiment_confidence", "sentiment_label",
    ]

    # Fallback path for local/offline recovery when Flink sinks are empty.
    if users_raw.empty:
        users_secret = read_secret_csv("users.csv")
        if not users_secret.empty:
            users_secret = users_secret.rename(columns={"id": "user_id"})
            users_raw = align_columns(users_secret, users_cols)
    if sessions_raw.empty:
        sessions_secret = read_secret_csv("sessions.csv")
        if not sessions_secret.empty:
            sessions_secret = sessions_secret.rename(columns={"id": "session_id"})
            sessions_raw = align_columns(sessions_secret, sessions_cols)
    if feedbacks_raw.empty:
        feedback_secret = read_secret_csv("feedbacks.csv")
        if not feedback_secret.empty:
            feedback_secret = feedback_secret.rename(columns={"id": "feedback_id"})
            feedbacks_raw = align_columns(feedback_secret, feedback_cols)
    if messages_raw.empty:
        messages_secret = read_secret_csv("whatsapp_messages.csv")
        if not messages_secret.empty:
            messages_secret = messages_secret.rename(columns={"id": "message_id"})
            
            # Prefer Step 1 bulk sentiment artifacts if available
            artifact_path = Path("artifacts/sentiment/sentiment_scores.csv")
            if artifact_path.exists():
                print(f"    [Fallback] Loading high-quality sentiment from Step 1 artifact: {artifact_path}")
                scores_df = pd.read_csv(artifact_path)
                # Align on message text and session/time if IDs are risky, but typically 'id' works
                if "id" in scores_df.columns and "message_id" in messages_secret.columns:
                    messages_secret = messages_secret.merge(
                        scores_df[["id", "sentiment_score", "sentiment_confidence", "sentiment_label"]].rename(columns={"id": "message_id"}),
                        on="message_id", how="left"
                    )
                else:
                    # Fallback to heuristic if merge keys are missing
                    artifact_path = None

            if not artifact_path or messages_secret["sentiment_score"].isna().all():
                print("    [Fallback] High-quality artifact missing or unusable; falling back to heuristic sentiment.")
                import re
                def _heuristic(text: str) -> float:
                    s = str(text or "").strip().lower()
                    if not s: return 0.0
                    tokens = re.findall(r"[a-z']+", s)
                    if not tokens: return 0.0
                    pos_hits = sum(1 for t in tokens if t in {"good", "great", "awesome", "nice", "love", "happy", "thanks", "thankyou", "resolved", "perfect", "excellent", "fast", "smooth"})
                    neg_hits = sum(1 for t in tokens if t in {"bad", "worse", "worst", "hate", "angry", "upset", "frustrated", "annoyed", "terrible", "awful", "slow", "broken", "error", "issue", "problem", "failed"})
                    raw = (pos_hits - neg_hits) / max(len(tokens), 6)
                    if "!" in s: raw *= 1.1
                    if any(w in s for w in ["not good", "not happy", "never again"]): raw -= 0.2
                    if any(w in s for w in ["not bad", "works now", "all good"]): raw += 0.2
                    return float(max(min(raw * 2.0, 1.0), -1.0))
                
                messages_secret["sentiment_score"] = messages_secret["message"].apply(_heuristic)
                messages_secret["sentiment_confidence"] = 0.5
                messages_secret["sentiment_label"] = messages_secret["sentiment_score"].apply(lambda x: "positive" if x > 0.1 else ("negative" if x < -0.1 else "neutral"))
            
            messages_raw = align_columns(messages_secret, messages_cols)


    users_raw = normalize_raw_table(users_raw, users_cols)
    sessions_raw = normalize_raw_table(sessions_raw, sessions_cols)
    feedbacks_raw = normalize_raw_table(feedbacks_raw, feedback_cols)
    messages_raw = normalize_raw_table(messages_raw, messages_cols)

    # Users node table
    users = users_raw.copy()
    if users.empty:
        users_nodes = pd.DataFrame(columns=[
            "user_id", "created_at", "updated_at", "deleted_at", "first_name", "last_name", "full_name",
            "timezone", "country", "status", "type", "longitude", "latitude", "has_geo", "contacts_backfilled",
        ])
    else:
        users["user_id"] = pd.to_numeric(users["user_id"], errors="coerce").astype("Int64")
        users = users.dropna(subset=["user_id"]).drop_duplicates(subset=["user_id"], keep="last")
        users["user_id"] = users["user_id"].astype("int64")
        users["first_name"] = users["first_name"].fillna("").astype(str).str.strip()
        users["last_name"] = users["last_name"].fillna("").astype(str).str.strip()
        users["full_name"] = (users["first_name"] + " " + users["last_name"]).str.strip()
        users["timezone"] = users["timezone"].fillna("")
        users["country"] = users["country"].fillna("")
        users["status"] = users["status"].fillna("").astype(str).str.lower()
        users["type"] = users["type"].fillna("").astype(str).str.lower()
        users["longitude"] = pd.to_numeric(users["longitude"], errors="coerce")
        users["latitude"] = pd.to_numeric(users["latitude"], errors="coerce")
        users["has_geo"] = users["longitude"].notna() & users["latitude"].notna()
        users["contacts_backfilled"] = users["contacts_backfilled"].fillna(False).astype(str).isin(["t", "true", "1", "True"])
        users_nodes = users[
            [
                "user_id", "created_at", "updated_at", "deleted_at", "first_name", "last_name", "full_name",
                "timezone", "country", "status", "type", "longitude", "latitude", "has_geo", "contacts_backfilled",
            ]
        ]

    valid_user_ids = set(users_nodes["user_id"].tolist())

    # Sessions node table
    sess = sessions_raw.copy()
    if sess.empty:
        sessions_nodes = pd.DataFrame(columns=[
            "session_id", "user_id", "created_at", "updated_at", "deleted_at", "duration", "billed_duration",
            "provider", "has_transcription", "has_summary",
        ])
    else:
        sess["session_id"] = pd.to_numeric(sess["session_id"], errors="coerce").astype("Int64")
        sess["user_id"] = pd.to_numeric(sess["user_id"], errors="coerce").astype("Int64")
        sess = sess.dropna(subset=["session_id", "user_id"]).drop_duplicates(subset=["session_id"], keep="last")
        sess = sess[sess["user_id"].isin(valid_user_ids)]
        sess["session_id"] = sess["session_id"].astype("int64")
        sess["user_id"] = sess["user_id"].astype("int64")
        sess["duration"] = pd.to_numeric(sess["duration"], errors="coerce").fillna(0).clip(lower=0)
        sess["billed_duration"] = pd.to_numeric(sess["billed_duration"], errors="coerce").fillna(0).clip(lower=0)
        sess["provider"] = sess["provider"].fillna("").astype(str).str.lower()
        sess["has_transcription"] = sess["transcription"].fillna("").astype(str).str.strip().str.len() > 0
        sess["has_summary"] = sess["summary"].fillna("").astype(str).str.strip().str.len() > 0
        sessions_nodes = sess[
            [
                "session_id", "user_id", "created_at", "updated_at", "deleted_at", "duration", "billed_duration",
                "provider", "has_transcription", "has_summary",
            ]
        ]

    valid_session_ids = set(sessions_nodes["session_id"].tolist())

    # Feedback node table
    fb = feedbacks_raw.copy()
    if fb.empty:
        feedback_nodes = pd.DataFrame(columns=[
            "feedback_id", "user_id", "session_id", "created_at", "updated_at", "deleted_at", "feedback_source",
            "message", "feedback_char_len", "feedback_word_len",
        ])
    else:
        fb["feedback_id"] = pd.to_numeric(fb["feedback_id"], errors="coerce").astype("Int64")
        fb["user_id"] = pd.to_numeric(fb["user_id"], errors="coerce").astype("Int64")
        fb["session_id"] = pd.to_numeric(fb["session_id"], errors="coerce").astype("Int64")
        fb = fb.dropna(subset=["feedback_id", "user_id", "session_id"]).drop_duplicates(subset=["feedback_id"], keep="last")
        fb = fb[fb["user_id"].isin(valid_user_ids) & fb["session_id"].isin(valid_session_ids)]
        fb["feedback_id"] = fb["feedback_id"].astype("int64")
        fb["user_id"] = fb["user_id"].astype("int64")
        fb["session_id"] = fb["session_id"].astype("int64")
        fb["feedback_source"] = fb["feedback_source"].fillna("").astype(str).str.lower()
        fb["message"] = fb["message"].fillna("").astype(str)
        fb["feedback_char_len"] = fb["message"].str.len()
        fb["feedback_word_len"] = fb["message"].str.split().str.len().fillna(0).astype("int64")
        feedback_nodes = fb[
            [
                "feedback_id", "user_id", "session_id", "created_at", "updated_at", "deleted_at", "feedback_source",
                "message", "feedback_char_len", "feedback_word_len",
            ]
        ]

    # Messages node table (from Flink sentiment sink)
    msg = messages_raw.copy()
    if msg.empty:
        messages_nodes = pd.DataFrame(columns=[
            "message_id", "session_id", "created_at", "updated_at", "deleted_at", "role", "status", "model_name",
            "recipient_name", "message", "message_char_len", "message_word_len", "input_tokens", "output_tokens",
            "cost_usd", "sentiment_score", "sentiment_confidence", "sentiment_label",
        ])
    else:
        msg["message_id"] = pd.to_numeric(msg["message_id"], errors="coerce").astype("Int64")
        msg["session_id"] = pd.to_numeric(msg["session_id"], errors="coerce").astype("Int64")
        msg = msg.dropna(subset=["message_id", "session_id"]).drop_duplicates(subset=["message_id"], keep="last")
        msg = msg[msg["session_id"].isin(valid_session_ids)]
        msg["message_id"] = msg["message_id"].astype("int64")
        msg["session_id"] = msg["session_id"].astype("int64")
        msg["role"] = msg["role"].fillna("").astype(str).str.lower()
        msg["status"] = msg["status"].fillna("").astype(str).str.lower()
        msg["model_name"] = msg["model_name"].fillna("").astype(str).str.lower()
        msg["recipient_name"] = msg["recipient_name"].fillna("").astype(str)
        msg["message"] = msg["message"].fillna("").astype(str)
        msg["message_char_len"] = msg["message"].str.len()
        msg["message_word_len"] = msg["message"].str.split().str.len().fillna(0).astype("int64")
        msg["input_tokens"] = pd.to_numeric(msg["input_tokens"], errors="coerce").fillna(0)
        msg["output_tokens"] = pd.to_numeric(msg["output_tokens"], errors="coerce").fillna(0)
        msg["cost_usd"] = pd.to_numeric(msg["cost_usd"], errors="coerce").fillna(0)
        msg["sentiment_score"] = pd.to_numeric(msg["sentiment_score"], errors="coerce").fillna(0)
        msg["sentiment_confidence"] = pd.to_numeric(msg["sentiment_confidence"], errors="coerce").fillna(0)
        msg["sentiment_label"] = msg["sentiment_label"].fillna("neutral").astype(str).str.lower()
        messages_nodes = msg[
            [
                "message_id", "session_id", "created_at", "updated_at", "deleted_at", "role", "status", "model_name",
                "recipient_name", "message", "message_char_len", "message_word_len", "input_tokens", "output_tokens",
                "cost_usd", "sentiment_score", "sentiment_confidence", "sentiment_label",
            ]
        ]

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    save_artifact_df(users_nodes, "users_nodes", OUT_DIR / "users_nodes.csv", index=False)
    save_artifact_df(sessions_nodes, "sessions_nodes", OUT_DIR / "sessions_nodes.csv", index=False)
    save_artifact_df(messages_nodes, "messages_nodes", OUT_DIR / "messages_nodes.csv", index=False)
    save_artifact_df(feedback_nodes, "feedback_nodes", OUT_DIR / "feedback_nodes.csv", index=False)

    print("Saved node tables from Flink outputs to gnn_preprocessed/")
    print(f"  users_nodes.csv: {len(users_nodes):,}")
    print(f"  sessions_nodes.csv: {len(sessions_nodes):,}")
    print(f"  messages_nodes.csv: {len(messages_nodes):,}")
    print(f"  feedback_nodes.csv: {len(feedback_nodes):,}")


if __name__ == "__main__":
    main()
