"""
Train a hetero-style GNN for user behavior analysis and user-level feature importance.

Inputs (from gnn_preprocessed/):
  - users_nodes.csv
  - sessions_nodes.csv
  - messages_nodes.csv
  - feedback_nodes.csv

Outputs (to gnn_outputs/):
  - user_behaviour_scores.csv
  - user_feature_importance_global.csv
  - user_feature_importance_per_user.csv
  - user_embeddings.csv

Usage:
  /path/to/python train_user_behavior_gnn.py
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F

from app_config import EMBEDDINGS_ARTIFACT_DIR, GNN_OUTPUT_DIR, GNN_PREPROCESSED_DIR
from lib.online_store import load_artifact_df, save_artifact_df


INPUT_DIR = GNN_PREPROCESSED_DIR
OUTPUT_DIR = GNN_OUTPUT_DIR


def parse_dt(col: pd.Series) -> pd.Series:
    return pd.to_datetime(col, utc=True, errors="coerce")


def minmax(col: pd.Series) -> pd.Series:
    col = col.astype(float)
    cmin = np.nanmin(col.values)
    cmax = np.nanmax(col.values)
    if np.isclose(cmax - cmin, 0.0):
        return pd.Series(np.zeros(len(col), dtype=np.float32), index=col.index)
    out = (col - cmin) / (cmax - cmin)
    return out.fillna(0.0).astype(np.float32)


def humanize_feature_name(name: str) -> str:
    raw = str(name).strip()
    if not raw:
        return ""
    s = raw.replace("_", " ").strip()
    s = " ".join(part for part in s.split() if part)
    return s.title()


def build_user_table(users: pd.DataFrame, sessions: pd.DataFrame, messages: pd.DataFrame, feedback: pd.DataFrame) -> pd.DataFrame:
    users = users.copy()
    sessions = sessions.copy()
    messages = messages.copy()
    feedback = feedback.copy()

    users["created_at"] = parse_dt(users["created_at"])
    users["account_age_days"] = (pd.Timestamp.now(tz="UTC") - users["created_at"]).dt.total_seconds() / 86400.0
    users["account_age_days"] = users["account_age_days"].clip(lower=0).fillna(0)

    sessions["duration"] = pd.to_numeric(sessions["duration"], errors="coerce").fillna(0)
    sessions["billed_duration"] = pd.to_numeric(sessions["billed_duration"], errors="coerce").fillna(0)
    sessions_agg = sessions.groupby("user_id", as_index=False).agg(
        session_count=("session_id", "count"),
        session_duration_sum=("duration", "sum"),
        session_duration_mean=("duration", "mean"),
        billed_duration_sum=("billed_duration", "sum"),
        transcription_ratio=("has_transcription", "mean"),
        summary_ratio=("has_summary", "mean"),
    )

    session_user = sessions[["session_id", "user_id"]].drop_duplicates()
    messages = messages.merge(session_user, on="session_id", how="left")
    messages["message_word_len"] = pd.to_numeric(messages["message_word_len"], errors="coerce").fillna(0)
    messages["message_char_len"] = pd.to_numeric(messages["message_char_len"], errors="coerce").fillna(0)
    messages["input_tokens"] = pd.to_numeric(messages["input_tokens"], errors="coerce").fillna(0)
    messages["output_tokens"] = pd.to_numeric(messages["output_tokens"], errors="coerce").fillna(0)
    messages["cost_usd"] = pd.to_numeric(messages["cost_usd"], errors="coerce").fillna(0)
    messages_agg = messages.groupby("user_id", as_index=False).agg(
        message_count=("message_id", "count"),
        msg_word_len_mean=("message_word_len", "mean"),
        msg_char_len_mean=("message_char_len", "mean"),
        input_tokens_sum=("input_tokens", "sum"),
        output_tokens_sum=("output_tokens", "sum"),
        cost_usd_sum=("cost_usd", "sum"),
    )

    feedback["feedback_word_len"] = pd.to_numeric(feedback["feedback_word_len"], errors="coerce").fillna(0)
    feedback["feedback_char_len"] = pd.to_numeric(feedback["feedback_char_len"], errors="coerce").fillna(0)
    feedback_agg = feedback.groupby("user_id", as_index=False).agg(
        feedback_count=("feedback_id", "count"),
        feedback_word_len_mean=("feedback_word_len", "mean"),
        feedback_char_len_mean=("feedback_char_len", "mean"),
    )

    out = users.merge(sessions_agg, on="user_id", how="left")
    out = out.merge(messages_agg, on="user_id", how="left")
    out = out.merge(feedback_agg, on="user_id", how="left")

    numeric_fill_cols = [
        "session_count",
        "session_duration_sum",
        "session_duration_mean",
        "billed_duration_sum",
        "transcription_ratio",
        "summary_ratio",
        "message_count",
        "msg_word_len_mean",
        "msg_char_len_mean",
        "input_tokens_sum",
        "output_tokens_sum",
        "cost_usd_sum",
        "feedback_count",
        "feedback_word_len_mean",
        "feedback_char_len_mean",
    ]
    for c in numeric_fill_cols:
        out[c] = pd.to_numeric(out[c], errors="coerce").fillna(0)

    out["engagement_score"] = (
        out["message_count"] * 0.4
        + out["session_count"] * 0.3
        + out["session_duration_sum"] * 0.2
        + out["feedback_count"] * 0.1
    )

    return out


def build_feature_matrix(user_df: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    df = user_df.copy()

    bool_cols = ["contacts_backfilled"]
    for c in bool_cols:
        df[c] = df[c].fillna(False).astype(int)

    # Drop geographic attributes from embeddings to avoid location-driven latent factors.
    cat_cols = ["status", "type"]
    cat_enc = pd.get_dummies(df[cat_cols].fillna("unknown"), prefix=cat_cols, drop_first=False)

    numeric_cols = [
        "account_age_days",
        "contacts_backfilled",
        "session_count",
        "session_duration_sum",
        "session_duration_mean",
        "billed_duration_sum",
        "transcription_ratio",
        "summary_ratio",
        "message_count",
        "msg_word_len_mean",
        "msg_char_len_mean",
        "input_tokens_sum",
        "output_tokens_sum",
        "cost_usd_sum",
        "feedback_count",
        "feedback_word_len_mean",
        "feedback_char_len_mean",
    ]

    num = df[numeric_cols].copy()
    for c in num.columns:
        num[c] = minmax(pd.to_numeric(num[c], errors="coerce").fillna(0.0))

    feature_df = pd.concat([num, cat_enc], axis=1)
    feature_names = feature_df.columns.tolist()
    return feature_df.astype(np.float32), feature_names


def make_index_map(ids: pd.Series) -> dict[int, int]:
    return {int(v): i for i, v in enumerate(ids.tolist())}


def build_embedding_dimension_labels(
    emb: np.ndarray,
    emb_cols: list[str],
    user_feature_df: pd.DataFrame,
    user_feature_names: list[str],
) -> pd.DataFrame:
    if emb.size == 0 or len(emb_cols) == 0 or user_feature_df.empty:
        return pd.DataFrame(columns=["feature", "label", "anchor_feature", "anchor_feature_label", "abs_correlation"])

    E = np.asarray(emb, dtype=np.float64)
    F = np.asarray(user_feature_df[user_feature_names].values, dtype=np.float64)
    n = E.shape[0]
    if n < 2:
        return pd.DataFrame(columns=["feature", "label", "anchor_feature", "anchor_feature_label", "abs_correlation"])

    e_mean = E.mean(axis=0, keepdims=True)
    e_std = E.std(axis=0, keepdims=True)
    f_mean = F.mean(axis=0, keepdims=True)
    f_std = F.std(axis=0, keepdims=True)

    e_std[e_std == 0] = 1.0
    f_std[f_std == 0] = 1.0

    E_z = (E - e_mean) / e_std
    F_z = (F - f_mean) / f_std
    corr = np.abs((E_z.T @ F_z) / float(max(n - 1, 1)))

    rows = []
    for emb_idx, emb_name in enumerate(emb_cols):
        row = corr[emb_idx]
        feat_idx = int(np.argmax(row))
        anchor = user_feature_names[feat_idx]
        anchor_label = humanize_feature_name(anchor)
        strength = float(row[feat_idx])
        rows.append(
            {
                "feature": emb_name,
                "label": f"{emb_name} - {anchor_label}",
                "anchor_feature": anchor,
                "anchor_feature_label": anchor_label,
                "abs_correlation": strength,
            }
        )

    return pd.DataFrame(rows).sort_values("feature")


def aggregate_to_target(src_emb: torch.Tensor, target_index: torch.Tensor, target_size: int) -> torch.Tensor:
    out = torch.zeros((target_size, src_emb.shape[1]), device=src_emb.device)
    counts = torch.zeros((target_size, 1), device=src_emb.device)
    out.index_add_(0, target_index, src_emb)
    ones = torch.ones((target_index.shape[0], 1), device=src_emb.device)
    counts.index_add_(0, target_index, ones)
    return out / torch.clamp(counts, min=1.0)


class UserBehaviorGNN(nn.Module):
    def __init__(self, user_in: int, session_in: int, message_in: int, feedback_in: int, hidden: int = 64):
        super().__init__()
        self.user_proj = nn.Linear(user_in, hidden)
        self.session_proj = nn.Linear(session_in, hidden)
        self.message_proj = nn.Linear(message_in, hidden)
        self.feedback_proj = nn.Linear(feedback_in, hidden)

        self.fuse = nn.Sequential(
            nn.Linear(hidden * 4, hidden),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
        )
        self.classifier = nn.Linear(hidden, 1)

    def forward(
        self,
        user_x: torch.Tensor,
        session_x: torch.Tensor,
        message_x: torch.Tensor,
        feedback_x: torch.Tensor,
        session_to_user: torch.Tensor,
        message_to_user: torch.Tensor,
        feedback_to_user: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        user_h = F.relu(self.user_proj(user_x))
        session_h = F.relu(self.session_proj(session_x))
        message_h = F.relu(self.message_proj(message_x))
        feedback_h = F.relu(self.feedback_proj(feedback_x))

        n_users = user_h.shape[0]
        sess_agg = aggregate_to_target(session_h, session_to_user, n_users)
        msg_agg = aggregate_to_target(message_h, message_to_user, n_users)
        fb_agg = aggregate_to_target(feedback_h, feedback_to_user, n_users)

        fused = torch.cat([user_h, sess_agg, msg_agg, fb_agg], dim=1)
        user_emb = self.fuse(fused)
        logits = self.classifier(user_emb).squeeze(1)
        return logits, user_emb


def train_model(
    model: nn.Module,
    user_x: torch.Tensor,
    session_x: torch.Tensor,
    message_x: torch.Tensor,
    feedback_x: torch.Tensor,
    session_to_user: torch.Tensor,
    message_to_user: torch.Tensor,
    feedback_to_user: torch.Tensor,
    y: torch.Tensor,
    train_mask: torch.Tensor,
    epochs: int = 250,
) -> None:
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)

    pos_weight = ((y[train_mask] == 0).sum().float() / torch.clamp((y[train_mask] == 1).sum().float(), min=1.0)).detach()
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    model.train()
    for epoch in range(1, epochs + 1):
        optimizer.zero_grad()
        logits, _ = model(user_x, session_x, message_x, feedback_x, session_to_user, message_to_user, feedback_to_user)
        loss = criterion(logits[train_mask], y[train_mask])
        loss.backward()
        optimizer.step()

        if epoch % 50 == 0 or epoch == 1:
            with torch.no_grad():
                probs = torch.sigmoid(logits)
                preds = (probs > 0.5).float()
                acc = (preds[train_mask] == y[train_mask]).float().mean().item()
            print(f"epoch={epoch:03d} loss={loss.item():.4f} train_acc={acc:.4f}")


def build_train_mask(n: int, frac: float = 0.8, seed: int = 42) -> torch.Tensor:
    rng = np.random.default_rng(seed)
    idx = np.arange(n)
    rng.shuffle(idx)
    k = max(1, int(n * frac))
    mask = np.zeros(n, dtype=bool)
    mask[idx[:k]] = True
    return torch.tensor(mask)


def main() -> None:
    users = load_artifact_df("users_nodes", INPUT_DIR / "users_nodes.csv")
    sessions = load_artifact_df("sessions_nodes", INPUT_DIR / "sessions_nodes.csv")
    messages = load_artifact_df("messages_nodes", INPUT_DIR / "messages_nodes.csv")
    feedback = load_artifact_df("feedback_nodes", INPUT_DIR / "feedback_nodes.csv")

    user_df = build_user_table(users, sessions, messages, feedback)
    user_features_df, user_feature_names = build_feature_matrix(user_df)

    sessions_basic = sessions[["session_id", "user_id", "duration", "billed_duration", "has_transcription", "has_summary"]].copy()
    sessions_basic[["duration", "billed_duration"]] = sessions_basic[["duration", "billed_duration"]].apply(pd.to_numeric, errors="coerce").fillna(0)
    sessions_x = sessions_basic[["duration", "billed_duration", "has_transcription", "has_summary"]].astype(np.float32)

    messages_basic = messages[["message_id", "session_id", "message_word_len", "message_char_len", "input_tokens", "output_tokens", "cost_usd"]].copy()
    for c in ["message_word_len", "message_char_len", "input_tokens", "output_tokens", "cost_usd"]:
        messages_basic[c] = pd.to_numeric(messages_basic[c], errors="coerce").fillna(0)
    messages_x = messages_basic[["message_word_len", "message_char_len", "input_tokens", "output_tokens", "cost_usd"]].astype(np.float32)

    feedback_basic = feedback[["feedback_id", "user_id", "feedback_word_len", "feedback_char_len"]].copy()
    feedback_basic[["feedback_word_len", "feedback_char_len"]] = feedback_basic[["feedback_word_len", "feedback_char_len"]].apply(pd.to_numeric, errors="coerce").fillna(0)
    feedback_x = feedback_basic[["feedback_word_len", "feedback_char_len"]].astype(np.float32)

    user_index = make_index_map(user_df["user_id"])
    session_index = make_index_map(sessions_basic["session_id"])

    session_to_user = sessions_basic["user_id"].map(user_index).fillna(-1).astype(int)
    valid_session_mask = session_to_user >= 0
    sessions_x = sessions_x.loc[valid_session_mask].reset_index(drop=True)
    session_to_user = session_to_user.loc[valid_session_mask].reset_index(drop=True)

    msg_merge = messages_basic.merge(sessions_basic[["session_id", "user_id"]], on="session_id", how="left")
    message_to_user = msg_merge["user_id"].map(user_index).fillna(-1).astype(int)
    valid_message_mask = message_to_user >= 0
    messages_x = messages_x.loc[valid_message_mask].reset_index(drop=True)
    message_to_user = message_to_user.loc[valid_message_mask].reset_index(drop=True)

    feedback_to_user = feedback_basic["user_id"].map(user_index).fillna(-1).astype(int)
    valid_feedback_mask = feedback_to_user >= 0
    feedback_x = feedback_x.loc[valid_feedback_mask].reset_index(drop=True)
    feedback_to_user = feedback_to_user.loc[valid_feedback_mask].reset_index(drop=True)

    # Pseudo target: high-engagement user (top quartile).
    y_series = (user_df["engagement_score"] >= user_df["engagement_score"].quantile(0.75)).astype(np.float32)

    user_x_t = torch.tensor(user_features_df.values, dtype=torch.float32)
    session_x_t = torch.tensor(sessions_x.values, dtype=torch.float32)
    message_x_t = torch.tensor(messages_x.values, dtype=torch.float32)
    feedback_x_t = torch.tensor(feedback_x.values, dtype=torch.float32)
    y_t = torch.tensor(y_series.values, dtype=torch.float32)

    session_to_user_t = torch.tensor(session_to_user.values, dtype=torch.long)
    message_to_user_t = torch.tensor(message_to_user.values, dtype=torch.long)
    feedback_to_user_t = torch.tensor(feedback_to_user.values, dtype=torch.long)

    train_mask = build_train_mask(len(user_df), frac=0.8)

    model = UserBehaviorGNN(
        user_in=user_x_t.shape[1],
        session_in=session_x_t.shape[1],
        message_in=message_x_t.shape[1],
        feedback_in=feedback_x_t.shape[1],
        hidden=64,
    )

    train_model(
        model,
        user_x_t,
        session_x_t,
        message_x_t,
        feedback_x_t,
        session_to_user_t,
        message_to_user_t,
        feedback_to_user_t,
        y_t,
        train_mask,
        epochs=250,
    )

    model.eval()
    with torch.no_grad():
        logits, emb = model(
            user_x_t,
            session_x_t,
            message_x_t,
            feedback_x_t,
            session_to_user_t,
            message_to_user_t,
            feedback_to_user_t,
        )
        probs = torch.sigmoid(logits).cpu().numpy()
        pred = (probs > 0.5).astype(int)

    # Gradient-based per-user feature attribution on user feature inputs.
    user_x_grad = user_x_t.clone().detach().requires_grad_(True)
    logits_grad, _ = model(
        user_x_grad,
        session_x_t,
        message_x_t,
        feedback_x_t,
        session_to_user_t,
        message_to_user_t,
        feedback_to_user_t,
    )
    score = torch.sigmoid(logits_grad).sum()
    score.backward()
    attributions = (user_x_grad.grad * user_x_grad).abs().detach().cpu().numpy()

    global_imp = attributions.mean(axis=0)
    global_rank = pd.DataFrame({
        "feature": user_feature_names,
        "importance": global_imp,
    }).sort_values("importance", ascending=False)

    per_user_rows = []
    for i, uid in enumerate(user_df["user_id"].tolist()):
        row_imp = attributions[i]
        order = np.argsort(-row_imp)
        top_k = min(10, len(order))
        for r in range(top_k):
            j = int(order[r])
            per_user_rows.append(
                {
                    "user_id": int(uid),
                    "rank": r + 1,
                    "feature": user_feature_names[j],
                    "importance": float(row_imp[j]),
                    "predicted_high_engagement_prob": float(probs[i]),
                }
            )
    per_user_imp = pd.DataFrame(per_user_rows)

    emb_cols = {f"emb_{i}": emb[:, i].detach().cpu().numpy() for i in range(emb.shape[1])}
    emb_df = pd.DataFrame({"user_id": user_df["user_id"].astype(int), **emb_cols})
    emb_col_names = list(emb_cols.keys())
    emb_label_df = build_embedding_dimension_labels(
        emb=emb.detach().cpu().numpy(),
        emb_cols=emb_col_names,
        user_feature_df=user_features_df,
        user_feature_names=user_feature_names,
    )

    scores_df = pd.DataFrame(
        {
            "user_id": user_df["user_id"].astype(int),
            "engagement_score": user_df["engagement_score"].astype(float),
            "high_engagement_label": y_series.astype(int),
            "pred_high_engagement": pred,
            "pred_high_engagement_prob": probs,
        }
    )

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    save_artifact_df(scores_df, "user_behaviour_scores", OUTPUT_DIR / "user_behaviour_scores.csv", index=False)
    save_artifact_df(global_rank, "user_feature_importance_global", OUTPUT_DIR / "user_feature_importance_global.csv", index=False)
    save_artifact_df(per_user_imp, "user_feature_importance_per_user", OUTPUT_DIR / "user_feature_importance_per_user.csv", index=False)
    save_artifact_df(emb_df, "user_embeddings", OUTPUT_DIR / "user_embeddings.csv", index=False)
    save_artifact_df(emb_label_df, "embedding_dimension_labels", OUTPUT_DIR / "embedding_dimension_labels.csv", index=False)

    # Canonical artifact locations for embedding-related CSVs.
    save_artifact_df(emb_df, "user_embeddings", EMBEDDINGS_ARTIFACT_DIR / "user_embeddings.csv", index=False)
    save_artifact_df(emb_label_df, "embedding_dimension_labels", EMBEDDINGS_ARTIFACT_DIR / "embedding_dimension_labels.csv", index=False)

    print("Saved GNN outputs to:")
    print(f"  {OUTPUT_DIR}")
    print("Files:")
    print("  - user_behaviour_scores.csv")
    print("  - user_feature_importance_global.csv")
    print("  - user_feature_importance_per_user.csv")
    print("  - user_embeddings.csv")
    print("  - embedding_dimension_labels.csv")


if __name__ == "__main__":
    main()
