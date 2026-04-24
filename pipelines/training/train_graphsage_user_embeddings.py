"""
Train a PyTorch GraphSAGE model on users.csv and sessions.csv,
and export 64-dimensional user embeddings to user_embeddings.csv.

Default paths:
  - secret_data/users.csv
  - secret_data/sessions.csv
  - user_embeddings.csv

Usage:
  /path/to/python train_graphsage_user_embeddings.py
  /path/to/python train_graphsage_user_embeddings.py --users users.csv --sessions sessions.csv --out user_embeddings.csv
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F

from config import EMBEDDINGS_ARTIFACT_DIR, SECRET_DATA_DIR
from lib.device_utils import resolve_device
from lib.online_store import save_artifact_df


def resolve_input_path(path_str: str) -> Path:
    path = Path(path_str)
    if path.exists():
        return path

    if path.name.startswith("maya_"):
        alt_name = path.name.removeprefix("maya_")
    else:
        alt_name = f"maya_{path.name}"

    alt_path = path.with_name(alt_name)
    if alt_path.exists():
        return alt_path

    raise FileNotFoundError(f"Input CSV not found: {path} (also checked {alt_path.name})")


def minmax(col: pd.Series) -> pd.Series:
    vals = pd.to_numeric(col, errors="coerce").astype(float)
    vmin = np.nanmin(vals.values) if len(vals) else 0.0
    vmax = np.nanmax(vals.values) if len(vals) else 0.0
    if np.isclose(vmax - vmin, 0.0):
        return pd.Series(np.zeros(len(vals), dtype=np.float32), index=vals.index)
    return ((vals - vmin) / (vmax - vmin)).fillna(0.0).astype(np.float32)


def build_user_features(users: pd.DataFrame) -> tuple[pd.DataFrame, str]:
    if "id" in users.columns:
        user_id_col = "id"
    elif "user_id" in users.columns:
        user_id_col = "user_id"
    else:
        raise ValueError("users.csv must contain either 'id' or 'user_id' column")

    u = users.copy()
    u[user_id_col] = pd.to_numeric(u[user_id_col], errors="coerce")
    u = u.dropna(subset=[user_id_col]).copy()
    u[user_id_col] = u[user_id_col].astype(int)

    # Remove geographic fields to prevent location leakage into embeddings.
    geo_cols = {"longitude", "latitude", "country", "timezone", "city", "state", "region", "zip", "postal_code"}
    drop_cols = [c for c in u.columns if c.lower() in geo_cols]
    if drop_cols:
        u = u.drop(columns=drop_cols, errors="ignore")

    numeric_cols = []
    for c in u.columns:
        if c == user_id_col:
            continue
        if pd.api.types.is_numeric_dtype(u[c]):
            numeric_cols.append(c)

    bool_like = [c for c in ["contacts_backfilled"] if c in u.columns]
    for c in bool_like:
        u[c] = u[c].astype(str).str.lower().isin(["true", "t", "1", "yes", "y"]).astype(int)
        if c not in numeric_cols:
            numeric_cols.append(c)

    num_df = pd.DataFrame(index=u.index)
    for c in numeric_cols:
        num_df[c] = minmax(u[c])

    cat_cols = [c for c in ["status", "type"] if c in u.columns]
    cat_df = pd.DataFrame(index=u.index)
    if cat_cols:
        trimmed = u[cat_cols].fillna("unknown").astype(str)
        for c in cat_cols:
            top = trimmed[c].value_counts(dropna=False).head(25).index
            trimmed[c] = np.where(trimmed[c].isin(top), trimmed[c], "other")
        cat_df = pd.get_dummies(trimmed, prefix=cat_cols, drop_first=False)

    feats = pd.concat([num_df, cat_df], axis=1)
    if feats.shape[1] == 0:
        feats = pd.DataFrame({"bias": np.ones(len(u), dtype=np.float32)}, index=u.index)
    feats = feats.apply(pd.to_numeric, errors="coerce").fillna(0.0).astype(np.float32)

    out = pd.concat([u[[user_id_col]].reset_index(drop=True), feats.reset_index(drop=True)], axis=1)
    return out, user_id_col


def build_session_features(sessions: pd.DataFrame) -> tuple[pd.DataFrame, str, str]:
    if "id" not in sessions.columns:
        raise ValueError("sessions.csv must contain 'id' column")
    if "user_id" not in sessions.columns:
        raise ValueError("sessions.csv must contain 'user_id' column")

    s = sessions.copy()
    s["id"] = pd.to_numeric(s["id"], errors="coerce")
    s["user_id"] = pd.to_numeric(s["user_id"], errors="coerce")
    s = s.dropna(subset=["id", "user_id"]).copy()
    s["id"] = s["id"].astype(int)
    s["user_id"] = s["user_id"].astype(int)

    numeric_cols = []
    for c in ["duration", "billed_duration"]:
        if c in s.columns:
            numeric_cols.append(c)

    for c in ["transcription", "short_summary", "summary"]:
        if c in s.columns:
            s[f"has_{c}"] = s[c].fillna("").astype(str).str.strip().str.len().gt(0).astype(int)
            numeric_cols.append(f"has_{c}")

    num_df = pd.DataFrame(index=s.index)
    for c in numeric_cols:
        num_df[c] = minmax(s[c])

    cat_cols = [c for c in ["provider"] if c in s.columns]
    cat_df = pd.DataFrame(index=s.index)
    if cat_cols:
        trimmed = s[cat_cols].fillna("unknown").astype(str)
        for c in cat_cols:
            top = trimmed[c].value_counts(dropna=False).head(10).index
            trimmed[c] = np.where(trimmed[c].isin(top), trimmed[c], "other")
        cat_df = pd.get_dummies(trimmed, prefix=cat_cols, drop_first=False)

    feats = pd.concat([num_df, cat_df], axis=1)
    if feats.shape[1] == 0:
        feats = pd.DataFrame({"bias": np.ones(len(s), dtype=np.float32)}, index=s.index)
    feats = feats.apply(pd.to_numeric, errors="coerce").fillna(0.0).astype(np.float32)

    out = pd.concat([s[["id", "user_id"]].reset_index(drop=True), feats.reset_index(drop=True)], axis=1)
    return out, "id", "user_id"


def aggregate_mean(src_x: torch.Tensor, src_index: torch.Tensor, dst_size: int) -> torch.Tensor:
    out = torch.zeros((dst_size, src_x.shape[1]), dtype=src_x.dtype, device=src_x.device)
    cnt = torch.zeros((dst_size, 1), dtype=src_x.dtype, device=src_x.device)
    out.index_add_(0, src_index, src_x)
    ones = torch.ones((src_index.shape[0], 1), dtype=src_x.dtype, device=src_x.device)
    cnt.index_add_(0, src_index, ones)
    return out / torch.clamp(cnt, min=1.0)


class SageLayer(nn.Module):
    def __init__(self, in_dim: int, out_dim: int):
        super().__init__()
        self.lin_self = nn.Linear(in_dim, out_dim)
        self.lin_neigh = nn.Linear(in_dim, out_dim)

    def forward(self, self_x: torch.Tensor, neigh_agg: torch.Tensor) -> torch.Tensor:
        return F.relu(self.lin_self(self_x) + self.lin_neigh(neigh_agg))


class BipartiteGraphSAGE(nn.Module):
    def __init__(self, user_in: int, session_in: int, hidden: int = 128, out_dim: int = 64):
        super().__init__()
        self.user_in_proj = nn.Linear(user_in, hidden)
        self.session_in_proj = nn.Linear(session_in, hidden)

        self.user_sage_1 = SageLayer(hidden, hidden)
        self.session_sage_1 = SageLayer(hidden, hidden)
        self.user_sage_2 = SageLayer(hidden, hidden)
        self.session_sage_2 = SageLayer(hidden, hidden)

        self.user_out = nn.Linear(hidden, out_dim)
        self.session_out = nn.Linear(hidden, out_dim)

    def forward(
        self,
        user_x: torch.Tensor,
        session_x: torch.Tensor,
        edge_user_idx: torch.Tensor,
        edge_session_idx: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        u0 = F.relu(self.user_in_proj(user_x))
        s0 = F.relu(self.session_in_proj(session_x))

        u_neigh_1 = aggregate_mean(s0[edge_session_idx], edge_user_idx, u0.shape[0])
        s_neigh_1 = aggregate_mean(u0[edge_user_idx], edge_session_idx, s0.shape[0])

        u1 = self.user_sage_1(u0, u_neigh_1)
        s1 = self.session_sage_1(s0, s_neigh_1)

        u_neigh_2 = aggregate_mean(s1[edge_session_idx], edge_user_idx, u1.shape[0])
        s_neigh_2 = aggregate_mean(u1[edge_user_idx], edge_session_idx, s1.shape[0])

        u2 = self.user_sage_2(u1, u_neigh_2)
        s2 = self.session_sage_2(s1, s_neigh_2)

        user_emb = F.normalize(self.user_out(u2), p=2, dim=1)
        session_emb = F.normalize(self.session_out(s2), p=2, dim=1)
        return user_emb, session_emb


def train_link_prediction(
    model: BipartiteGraphSAGE,
    user_x: torch.Tensor,
    session_x: torch.Tensor,
    edge_u: torch.Tensor,
    edge_s: torch.Tensor,
    n_sessions: int,
    epochs: int = 120,
    lr: float = 1e-3,
) -> None:
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    bce = nn.BCEWithLogitsLoss()
    rng = np.random.default_rng(42)

    for ep in range(1, epochs + 1):
        model.train()
        opt.zero_grad()

        user_emb, session_emb = model(user_x, session_x, edge_u, edge_s)

        pos_logits = (user_emb[edge_u] * session_emb[edge_s]).sum(dim=1)
        pos_labels = torch.ones_like(pos_logits)

        neg_s_np = rng.integers(0, n_sessions, size=edge_s.shape[0], endpoint=False)
        neg_s = torch.tensor(neg_s_np, dtype=torch.long, device=edge_s.device)
        neg_logits = (user_emb[edge_u] * session_emb[neg_s]).sum(dim=1)
        neg_labels = torch.zeros_like(neg_logits)

        logits = torch.cat([pos_logits, neg_logits], dim=0)
        labels = torch.cat([pos_labels, neg_labels], dim=0)
        loss = bce(logits, labels)

        loss.backward()
        opt.step()

        if ep == 1 or ep % 20 == 0:
            with torch.no_grad():
                preds = (torch.sigmoid(logits) > 0.5).float()
                acc = (preds == labels).float().mean().item()
            print(f"epoch={ep:03d} loss={loss.item():.4f} acc={acc:.4f}")


def parse_args() -> argparse.Namespace:
    base = EMBEDDINGS_ARTIFACT_DIR
    p = argparse.ArgumentParser(description="Train GraphSAGE user embeddings from users.csv and sessions.csv")
    p.add_argument("--users", type=str, default=str(SECRET_DATA_DIR / "users.csv"))
    p.add_argument("--sessions", type=str, default=str(SECRET_DATA_DIR / "sessions.csv"))
    p.add_argument("--out", type=str, default=str(base / "user_embeddings.csv"))
    p.add_argument("--epochs", type=int, default=120)
    p.add_argument("--hidden", type=int, default=128)
    p.add_argument("--dim", type=int, default=64)
    p.add_argument("--lr", type=float, default=1e-3)
    return p.parse_args()


def main() -> None:
    args = parse_args()

    users_path = resolve_input_path(args.users)
    sessions_path = resolve_input_path(args.sessions)
    out_path = Path(args.out)

    users_raw = pd.read_csv(users_path)
    sessions_raw = pd.read_csv(sessions_path)

    users_df, user_id_col = build_user_features(users_raw)
    sessions_df, session_id_col, session_user_col = build_session_features(sessions_raw)

    user_ids = users_df[user_id_col].astype(int).tolist()
    session_ids = sessions_df[session_id_col].astype(int).tolist()

    user_idx = {uid: i for i, uid in enumerate(user_ids)}
    session_idx = {sid: i for i, sid in enumerate(session_ids)}

    edge_df = sessions_df[[session_id_col, session_user_col]].copy()
    edge_df["u"] = edge_df[session_user_col].map(user_idx)
    edge_df["s"] = edge_df[session_id_col].map(session_idx)
    edge_df = edge_df.dropna(subset=["u", "s"]).copy()
    edge_df["u"] = edge_df["u"].astype(int)
    edge_df["s"] = edge_df["s"].astype(int)

    if edge_df.empty:
        raise ValueError("No valid user-session edges found after ID mapping.")

    user_feat_cols = [c for c in users_df.columns if c != user_id_col]
    session_feat_cols = [c for c in sessions_df.columns if c not in [session_id_col, session_user_col]]

    user_x = torch.tensor(users_df[user_feat_cols].values, dtype=torch.float32)
    session_x = torch.tensor(sessions_df[session_feat_cols].values, dtype=torch.float32)
    edge_u = torch.tensor(edge_df["u"].values, dtype=torch.long)
    edge_s = torch.tensor(edge_df["s"].values, dtype=torch.long)

    device = resolve_device()

    model = BipartiteGraphSAGE(
        user_in=user_x.shape[1],
        session_in=session_x.shape[1],
        hidden=args.hidden,
        out_dim=args.dim,
    ).to(device)

    user_x = user_x.to(device)
    session_x = session_x.to(device)
    edge_u = edge_u.to(device)
    edge_s = edge_s.to(device)

    model_path = EMBEDDINGS_ARTIFACT_DIR / "graphsage_model.pt"
    if model_path.exists():
        print(f"Loading existing GraphSAGE model from {model_path}")
        model.load_state_dict(torch.load(model_path, map_location=device))
    else:
        train_link_prediction(
            model,
            user_x,
            session_x,
            edge_u,
            edge_s,
            n_sessions=session_x.shape[0],
            epochs=args.epochs,
            lr=args.lr,
        )

    torch.save(model.state_dict(), EMBEDDINGS_ARTIFACT_DIR / "graphsage_model.pt")

    model.eval()
    with torch.no_grad():
        user_emb, _ = model(user_x, session_x, edge_u, edge_s)

    emb_np = user_emb.detach().cpu().numpy()
    out_df = pd.DataFrame({"user_id": user_ids})
    for i in range(emb_np.shape[1]):
        out_df[f"emb_{i}"] = emb_np[:, i]

    save_artifact_df(out_df, "user_embeddings", out_path, index=False)
    print(f"Saved {emb_np.shape[0]} user embeddings ({emb_np.shape[1]}-d) to: {out_path}")


if __name__ == "__main__":
    main()
