"""
GNN Graph Builder — Heterogeneous Graph from Maya CSV Data

Constructs a PyTorch Geometric HeteroData graph with:
  Node types: User, Session, Message, Feedback
  Edge types: User↔Session, Session↔Message, User↔Feedback,
              Session↔Feedback, Message→Message (temporal)

Usage:
    source flink_venv/bin/activate
    python gnn_graph_builder.py
"""

import pandas as pd
import numpy as np
from pathlib import Path
from textblob import TextBlob
import torch
from torch_geometric.data import HeteroData
import warnings
import re

warnings.filterwarnings("ignore")

# ── Configuration ────────────────────────────────────────────────────────────

DATA_DIR = Path("/Users/arthxoxo/maya_targeted_backups")
OUTPUT_DIR = Path("/Users/arthxoxo/maya-ML-backend")
FEATURE_MATRIX_PATH = OUTPUT_DIR / "user_feature_matrix.csv"

# ── Data Loading ─────────────────────────────────────────────────────────────


def load_raw_data():
    """Load all CSVs with proper date parsing."""
    print("📂  Loading raw CSVs...")

    users = pd.read_csv(DATA_DIR / "users.csv")
    users["created_at"] = pd.to_datetime(
        users["created_at"], format="mixed", utc=True, errors="coerce"
    )

    messages = pd.read_csv(DATA_DIR / "whatsapp_messages.csv")
    messages["created_at"] = pd.to_datetime(
        messages["created_at"], format="mixed", utc=True, errors="coerce"
    )

    sessions = pd.read_csv(DATA_DIR / "sessions.csv")
    sessions["created_at"] = pd.to_datetime(
        sessions["created_at"], format="mixed", utc=True, errors="coerce"
    )

    feedbacks = pd.read_csv(DATA_DIR / "feedbacks.csv")
    feedbacks["created_at"] = pd.to_datetime(
        feedbacks["created_at"], format="mixed", utc=True, errors="coerce"
    )

    print(f"    Users:    {len(users):>7,}")
    print(f"    Messages: {len(messages):>7,}")
    print(f"    Sessions: {len(sessions):>7,}")
    print(f"    Feedback: {len(feedbacks):>7,}")

    return users, messages, sessions, feedbacks


# ── Sentiment Helpers ────────────────────────────────────────────────────────


def get_sentiment(text: str) -> float:
    if not isinstance(text, str) or not text.strip():
        return 0.0
    return round(TextBlob(text).sentiment.polarity, 4)


def get_subjectivity(text: str) -> float:
    if not isinstance(text, str) or not text.strip():
        return 0.0
    return round(TextBlob(text).sentiment.subjectivity, 4)


# ── Node Feature Builders ───────────────────────────────────────────────────


def build_user_node_features(users, feature_matrix):
    """
    Build user node features from the pre-computed feature matrix.
    Returns: feature tensor, user_id → node_index mapping
    """
    print("\n🔧  Building User node features...")

    # Merge user identity with feature matrix
    fm = feature_matrix.copy()

    # Build id → index mapping
    user_ids = fm["user_id"].tolist()
    user_id_to_idx = {uid: idx for idx, uid in enumerate(user_ids)}

    # Select numeric feature columns
    exclude_cols = ["user_id", "user_name", "first_name", "last_name",
                    "status", "type", "timezone"]
    feature_cols = [c for c in fm.columns if c not in exclude_cols
                    and fm[c].dtype in [np.float64, np.int64, np.float32]]

    # Extract features and handle NaN
    features = fm[feature_cols].fillna(0).values.astype(np.float32)

    # Normalize features (zero-mean, unit-variance)
    mean = features.mean(axis=0, keepdims=True)
    std = features.std(axis=0, keepdims=True)
    std[std == 0] = 1.0
    features = (features - mean) / std

    print(f"    ✅ {len(user_ids)} User nodes × {features.shape[1]} features")

    return (
        torch.tensor(features, dtype=torch.float),
        user_id_to_idx,
        feature_cols,
    )


def build_session_node_features(sessions, user_id_to_idx):
    """
    Build session node features.
    Returns: feature tensor, session_id → node_index mapping
    """
    print("🔧  Building Session node features...")

    # Only include sessions for users we have in the graph
    valid_sessions = sessions[sessions["user_id"].isin(user_id_to_idx.keys())].copy()
    valid_sessions = valid_sessions.drop_duplicates(subset=["id"])
    valid_sessions = valid_sessions.sort_values("id").reset_index(drop=True)

    session_ids = valid_sessions["id"].tolist()
    session_id_to_idx = {sid: idx for idx, sid in enumerate(session_ids)}

    # Build features
    features = []
    for _, row in valid_sessions.iterrows():
        duration = float(row.get("duration", 0) or 0)
        has_transcription = 1.0 if pd.notna(row.get("transcription")) and str(row.get("transcription", "")).strip() else 0.0
        has_summary = 1.0 if pd.notna(row.get("summary")) and str(row.get("summary", "")).strip() else 0.0

        # Temporal features from created_at
        created = row.get("created_at")
        if pd.notna(created):
            hour = created.hour / 23.0  # Normalize to [0, 1]
            day_of_week = created.dayofweek / 6.0
        else:
            hour = 0.0
            day_of_week = 0.0

        features.append([
            duration,
            has_transcription,
            has_summary,
            hour,
            day_of_week,
        ])

    features = np.array(features, dtype=np.float32)

    # Normalize duration (leave binary/normalized features as-is)
    if features.shape[0] > 0:
        dur_mean = features[:, 0].mean()
        dur_std = features[:, 0].std()
        if dur_std > 0:
            features[:, 0] = (features[:, 0] - dur_mean) / dur_std

    print(f"    ✅ {len(session_ids)} Session nodes × {features.shape[1]} features")

    return (
        torch.tensor(features, dtype=torch.float),
        session_id_to_idx,
        valid_sessions,
    )


def build_message_node_features(messages, session_id_to_idx):
    """
    Build message node features.
    Returns: feature tensor, message_id → node_index mapping
    """
    print("🔧  Building Message node features...")

    # Only messages in valid sessions
    valid_msgs = messages[messages["session_id"].isin(session_id_to_idx.keys())].copy()
    valid_msgs = valid_msgs.drop_duplicates(subset=["id"])
    valid_msgs = valid_msgs.sort_values(["session_id", "created_at"]).reset_index(drop=True)

    msg_ids = valid_msgs["id"].tolist()
    msg_id_to_idx = {mid: idx for idx, mid in enumerate(msg_ids)}

    # Compute per-message features
    print("    Computing message-level sentiment (this may take a moment)...")

    features = []
    for _, row in valid_msgs.iterrows():
        text = str(row.get("message", "") or "")

        # Sentiment
        sentiment = get_sentiment(text)
        subjectivity = get_subjectivity(text)

        # Text length (normalized later)
        msg_len = len(text)

        # Is question
        question_words = r"^(what|how|why|when|where|who|which|can|could|would|is|are|do|does|did|will|shall)\b"
        is_question = 1.0 if (text.strip().endswith("?") or
                              bool(re.match(question_words, text.lower()))) else 0.0

        # Role encoding: user=1, assistant=0, other=0.5
        role = row.get("role", "")
        if role == "user":
            role_enc = 1.0
        elif role == "assistant":
            role_enc = 0.0
        else:
            role_enc = 0.5

        # Token counts (from assistant responses)
        input_tokens = float(row.get("input_tokens", 0) or 0)
        output_tokens = float(row.get("output_tokens", 0) or 0)

        # Temporal
        created = row.get("created_at")
        if pd.notna(created):
            hour = created.hour / 23.0
        else:
            hour = 0.0

        features.append([
            sentiment,
            subjectivity,
            msg_len,
            is_question,
            role_enc,
            input_tokens,
            output_tokens,
            hour,
        ])

    features = np.array(features, dtype=np.float32)

    # Normalize non-binary columns (msg_len, input_tokens, output_tokens)
    for col_idx in [2, 5, 6]:
        col = features[:, col_idx]
        col_mean = col.mean()
        col_std = col.std()
        if col_std > 0:
            features[:, col_idx] = (col - col_mean) / col_std

    print(f"    ✅ {len(msg_ids)} Message nodes × {features.shape[1]} features")

    return (
        torch.tensor(features, dtype=torch.float),
        msg_id_to_idx,
        valid_msgs,
    )


def build_feedback_node_features(feedbacks, user_id_to_idx):
    """
    Build feedback node features.
    Returns: feature tensor, feedback_id → node_index mapping
    """
    print("🔧  Building Feedback node features...")

    valid_fb = feedbacks[feedbacks["user_id"].isin(user_id_to_idx.keys())].copy()
    valid_fb = valid_fb.drop_duplicates(subset=["id"])
    valid_fb = valid_fb.sort_values("id").reset_index(drop=True)

    fb_ids = valid_fb["id"].tolist()
    fb_id_to_idx = {fid: idx for idx, fid in enumerate(fb_ids)}

    features = []
    for _, row in valid_fb.iterrows():
        text = str(row.get("message", "") or "")
        sentiment = get_sentiment(text)
        word_count = len(text.split())
        subjectivity = get_subjectivity(text)

        features.append([
            sentiment,
            word_count,
            subjectivity,
        ])

    features = np.array(features, dtype=np.float32)

    # Normalize word count
    if features.shape[0] > 0:
        wc = features[:, 1]
        wc_mean = wc.mean()
        wc_std = wc.std()
        if wc_std > 0:
            features[:, 1] = (wc - wc_mean) / wc_std

    print(f"    ✅ {len(fb_ids)} Feedback nodes × {features.shape[1]} features")

    return (
        torch.tensor(features, dtype=torch.float),
        fb_id_to_idx,
        valid_fb,
    )


# ── Edge Builders ────────────────────────────────────────────────────────────


def build_edges(users_df, sessions_df, messages_df, feedbacks_df,
                user_id_to_idx, session_id_to_idx, msg_id_to_idx, fb_id_to_idx,
                valid_sessions, valid_msgs, valid_fb):
    """Build all edge types for the heterogeneous graph."""
    print("\n🔗  Building edges...")

    edges = {}

    # ── User ↔ Session ───────────────────────────────────────────────────
    u2s_src, u2s_dst = [], []
    for _, row in valid_sessions.iterrows():
        uid = row["user_id"]
        sid = row["id"]
        if uid in user_id_to_idx and sid in session_id_to_idx:
            u2s_src.append(user_id_to_idx[uid])
            u2s_dst.append(session_id_to_idx[sid])

    if u2s_src:
        edges[("user", "has_session", "session")] = torch.tensor(
            [u2s_src, u2s_dst], dtype=torch.long
        )
        edges[("session", "belongs_to", "user")] = torch.tensor(
            [u2s_dst, u2s_src], dtype=torch.long
        )
    print(f"    User ↔ Session: {len(u2s_src)} edges")

    # ── Session ↔ Message ────────────────────────────────────────────────
    s2m_src, s2m_dst = [], []
    for _, row in valid_msgs.iterrows():
        sid = row["session_id"]
        mid = row["id"]
        if sid in session_id_to_idx and mid in msg_id_to_idx:
            s2m_src.append(session_id_to_idx[sid])
            s2m_dst.append(msg_id_to_idx[mid])

    if s2m_src:
        edges[("session", "contains", "message")] = torch.tensor(
            [s2m_src, s2m_dst], dtype=torch.long
        )
        edges[("message", "in_session", "session")] = torch.tensor(
            [s2m_dst, s2m_src], dtype=torch.long
        )
    print(f"    Session ↔ Message: {len(s2m_src)} edges")

    # ── User ↔ Feedback ──────────────────────────────────────────────────
    u2f_src, u2f_dst = [], []
    for _, row in valid_fb.iterrows():
        uid = row["user_id"]
        fid = row["id"]
        if uid in user_id_to_idx and fid in fb_id_to_idx:
            u2f_src.append(user_id_to_idx[uid])
            u2f_dst.append(fb_id_to_idx[fid])

    if u2f_src:
        edges[("user", "gave_feedback", "feedback")] = torch.tensor(
            [u2f_src, u2f_dst], dtype=torch.long
        )
        edges[("feedback", "from_user", "user")] = torch.tensor(
            [u2f_dst, u2f_src], dtype=torch.long
        )
    print(f"    User ↔ Feedback: {len(u2f_src)} edges")

    # ── Session ↔ Feedback ───────────────────────────────────────────────
    s2f_src, s2f_dst = [], []
    for _, row in valid_fb.iterrows():
        sid = row.get("session_id")
        fid = row["id"]
        if pd.notna(sid) and int(sid) in session_id_to_idx and fid in fb_id_to_idx:
            s2f_src.append(session_id_to_idx[int(sid)])
            s2f_dst.append(fb_id_to_idx[fid])

    if s2f_src:
        edges[("session", "has_feedback", "feedback")] = torch.tensor(
            [s2f_src, s2f_dst], dtype=torch.long
        )
        edges[("feedback", "about_session", "session")] = torch.tensor(
            [s2f_dst, s2f_src], dtype=torch.long
        )
    print(f"    Session ↔ Feedback: {len(s2f_src)} edges")

    # ── Message → Message (temporal within same session) ─────────────────
    m2m_src, m2m_dst = [], []
    for sid, grp in valid_msgs.groupby("session_id"):
        grp_sorted = grp.sort_values("created_at")
        grp_indices = [msg_id_to_idx[mid] for mid in grp_sorted["id"] if mid in msg_id_to_idx]
        for i in range(len(grp_indices) - 1):
            m2m_src.append(grp_indices[i])
            m2m_dst.append(grp_indices[i + 1])

    if m2m_src:
        edges[("message", "followed_by", "message")] = torch.tensor(
            [m2m_src, m2m_dst], dtype=torch.long
        )
    print(f"    Message → Message (temporal): {len(m2m_src)} edges")

    return edges


# ── Main Builder ─────────────────────────────────────────────────────────────


def build_graph():
    """Build the complete heterogeneous graph."""
    print("=" * 60)
    print("  Maya ML — GNN Graph Construction Pipeline")
    print("=" * 60)

    # ── Load data ────────────────────────────────────────────────────────
    users, messages, sessions, feedbacks = load_raw_data()
    feature_matrix = pd.read_csv(FEATURE_MATRIX_PATH)

    # ── Build node features ──────────────────────────────────────────────
    user_features, user_id_to_idx, user_feature_cols = build_user_node_features(
        users, feature_matrix
    )

    session_features, session_id_to_idx, valid_sessions = build_session_node_features(
        sessions, user_id_to_idx
    )

    message_features, msg_id_to_idx, valid_msgs = build_message_node_features(
        messages, session_id_to_idx
    )

    feedback_features, fb_id_to_idx, valid_fb = build_feedback_node_features(
        feedbacks, user_id_to_idx
    )

    # ── Build edges ──────────────────────────────────────────────────────
    edge_dict = build_edges(
        users, valid_sessions, valid_msgs, valid_fb,
        user_id_to_idx, session_id_to_idx, msg_id_to_idx, fb_id_to_idx,
        valid_sessions, valid_msgs, valid_fb,
    )

    # ── Assemble HeteroData ──────────────────────────────────────────────
    print("\n📦  Assembling HeteroData graph...")

    data = HeteroData()

    # Node features
    data["user"].x = user_features
    data["session"].x = session_features
    data["message"].x = message_features
    data["feedback"].x = feedback_features

    # Edge indices
    for edge_type, edge_index in edge_dict.items():
        data[edge_type].edge_index = edge_index

    # ── Target labels (engagement_score) ─────────────────────────────────
    engagement = feature_matrix.set_index("user_id")["engagement_score"]
    user_ids_ordered = [uid for uid, _ in sorted(user_id_to_idx.items(), key=lambda x: x[1])]
    labels = torch.tensor(
        [float(engagement.get(uid, 0.0)) for uid in user_ids_ordered],
        dtype=torch.float,
    )
    data["user"].y = labels

    # ── Store metadata ───────────────────────────────────────────────────
    data.user_feature_cols = user_feature_cols
    data.user_id_to_idx = user_id_to_idx
    data.session_id_to_idx = session_id_to_idx

    # Store user names for dashboard
    user_names = feature_matrix.set_index("user_id")["user_name"]
    data.user_names = [str(user_names.get(uid, "Unknown")) for uid in user_ids_ordered]

    # ── Save ─────────────────────────────────────────────────────────────
    output_path = OUTPUT_DIR / "graph_data.pt"
    torch.save(data, output_path)

    # ── Summary ──────────────────────────────────────────────────────────
    print(f"\n{'═' * 60}")
    print(f"✅  Heterogeneous graph saved: {output_path}")
    print(f"{'═' * 60}")
    print(f"\n📊  Graph Summary:")
    print(f"    Node Types:")
    for node_type in data.node_types:
        x = data[node_type].x
        print(f"      {node_type:12s}  {x.shape[0]:>6,} nodes × {x.shape[1]} features")
    print(f"    Edge Types:")
    for edge_type in data.edge_types:
        ei = data[edge_type].edge_index
        print(f"      {str(edge_type):65s}  {ei.shape[1]:>7,} edges")
    print(f"\n    Target: engagement_score on User nodes")
    print(f"    Label range: [{labels.min():.4f}, {labels.max():.4f}]")
    print(f"    Label mean:  {labels.mean():.4f}")

    return data


if __name__ == "__main__":
    build_graph()
