"""
Train a GRU model over per-user WhatsApp sentiment sequences to estimate mood swings.

Inputs:
  - sentiment_scores.csv (must include user-level message sentiment)

Outputs:
  - gru_mood_swing_summary.csv
  - gru_mood_training_report.csv
"""

from __future__ import annotations

import argparse
import random
from pathlib import Path

import numpy as np
import pandas as pd

from app_config import BASE_DIR, GNN_PREPROCESSED_DIR, SECRET_DATA_DIR, SENTIMENT_ARTIFACT_DIR
from lib.online_store import load_artifact_df, save_artifact_df


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

    return path


def write_empty_outputs(
    out_summary: Path,
    out_report: Path,
    total_messages: int,
    eligible_users: int,
    args: argparse.Namespace,
    warning: str,
) -> None:
    summary = pd.DataFrame(
        columns=[
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
    )
    report = pd.DataFrame(
        [
            {
                "total_messages": int(total_messages),
                "eligible_users": int(eligible_users),
                "sequence_length": int(args.sequence_length),
                "hidden_size": int(args.hidden_size),
                "epochs": int(args.epochs),
                "batch_size": int(args.batch_size),
                "train_samples": 0,
                "val_samples": 0,
                "train_loss": np.nan,
                "val_mse": np.nan,
                "val_baseline_mse": np.nan,
                "val_mse_vs_baseline_pct": np.nan,
                "validation_split_strategy": "not_run",
                "warning": warning,
            }
        ]
    )

    save_artifact_df(summary, "gru_mood_swing_summary", out_summary, index=False)
    save_artifact_df(report, "gru_mood_training_report", out_report, index=False)
    print(f"[warning] {warning}")
    print(f"[ok] Saved mood swing summary: {out_summary}")
    print(f"[ok] Saved training report: {out_report}")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train GRU mood swing model on WhatsApp/user sentiment timeline")
    p.add_argument("--sentiment", type=str, default=str(SENTIMENT_ARTIFACT_DIR / "sentiment_scores.csv"))
    p.add_argument("--sessions", type=str, default=str(SECRET_DATA_DIR / "sessions.csv"))
    p.add_argument("--out_summary", type=str, default=str(SENTIMENT_ARTIFACT_DIR / "gru_mood_swing_summary.csv"))
    p.add_argument("--out_report", type=str, default=str(SENTIMENT_ARTIFACT_DIR / "gru_mood_training_report.csv"))
    p.add_argument("--sequence_length", type=int, default=8)
    p.add_argument("--hidden_size", type=int, default=32)
    p.add_argument("--epochs", type=int, default=25)
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--learning_rate", type=float, default=1e-3)
    p.add_argument("--min_user_msgs", type=int, default=12)
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def _set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)


def _load_sentiment_messages(sentiment_path: Path, sessions_path: Path) -> pd.DataFrame:
    candidates = [
        sentiment_path,
        SENTIMENT_ARTIFACT_DIR / "sentiment_scores.csv",
        GNN_PREPROCESSED_DIR / "messages_nodes.csv",
        BASE_DIR / "sentiment_scores.csv",
    ]
    selected_path = next((p for p in candidates if p.exists()), None)

    if selected_path is not None:
        s = pd.read_csv(selected_path)
    else:
        s = load_artifact_df(
            artifact_key="sentiment_scores",
            fallback_csv_path=sentiment_path,
            required=False,
        )
        if s.empty:
            tried = ", ".join(str(p) for p in candidates)
            raise FileNotFoundError(
                "Sentiment file not found. Tried paths: "
                f"{tried}. Also checked Redis artifact key: sentiment_scores."
            )
    if "role" in s.columns:
        role = s["role"].astype(str).str.lower().str.strip()
        if role.eq("user").any():
            s = s[role.eq("user")].copy()

    if "user_id" not in s.columns:
        if "session_id" in s.columns and sessions_path.exists():
            sess = pd.read_csv(sessions_path, usecols=["id", "user_id"])
            sess["id"] = pd.to_numeric(sess["id"], errors="coerce")
            sess["user_id"] = pd.to_numeric(sess["user_id"], errors="coerce")
            s["session_id"] = pd.to_numeric(s["session_id"], errors="coerce")
            s = s.merge(sess.rename(columns={"id": "session_id"}), on="session_id", how="left")
        else:
            raise ValueError("sentiment_scores.csv must include user_id, or include session_id with sessions.csv available")

    s["user_id"] = pd.to_numeric(s.get("user_id"), errors="coerce")
    s = s.dropna(subset=["user_id"]).copy()
    s["user_id"] = s["user_id"].astype(int)
    if "sentiment_score" not in s.columns:
        lbl = s.get("sentiment_label", "neutral").fillna("neutral").astype(str).str.lower().str.strip()
        s["sentiment_score"] = np.where(lbl.eq("positive"), 0.25, np.where(lbl.eq("negative"), -0.25, 0.0))
    s["sentiment_score"] = pd.to_numeric(s.get("sentiment_score"), errors="coerce").fillna(0.0).clip(-1.0, 1.0)
    s["created_at"] = pd.to_datetime(s.get("created_at"), errors="coerce", utc=True)
    s["order_idx"] = np.arange(len(s))
    s = s.sort_values(["user_id", "created_at", "order_idx"], kind="mergesort").reset_index(drop=True)
    return s[["user_id", "created_at", "sentiment_score"]]


def _build_samples(
    df: pd.DataFrame,
    sequence_length: int,
    min_user_msgs: int,
) -> tuple[np.ndarray, np.ndarray, pd.DataFrame]:
    seq_x: list[np.ndarray] = []
    seq_y: list[float] = []
    meta_rows: list[dict] = []

    eligible_users = 0
    for user_id, grp in df.groupby("user_id", sort=False):
        values = grp["sentiment_score"].astype(float).to_numpy()
        if len(values) < max(min_user_msgs, sequence_length + 1):
            continue
        eligible_users += 1
        for idx in range(sequence_length, len(values)):
            seq_x.append(values[idx - sequence_length : idx].astype(np.float32))
            seq_y.append(float(values[idx]))
            meta_rows.append(
                {
                    "user_id": int(user_id),
                    "target_idx": int(idx),
                    "target_created_at": grp["created_at"].iloc[idx],
                }
            )

    if not seq_x:
        return np.zeros((0, sequence_length), dtype=np.float32), np.zeros((0,), dtype=np.float32), pd.DataFrame()
    x = np.stack(seq_x).astype(np.float32)
    y = np.array(seq_y, dtype=np.float32)
    meta = pd.DataFrame(meta_rows)
    meta["eligible_users"] = int(eligible_users)
    return x, y, meta


def _train_gru(
    x: np.ndarray,
    y: np.ndarray,
    meta: pd.DataFrame,
    hidden_size: int,
    epochs: int,
    batch_size: int,
    learning_rate: float,
    seed: int,
) -> tuple[object, float, float, float, int, int]:
    try:
        import torch
        import torch.nn as nn
        from torch.utils.data import DataLoader, TensorDataset
    except Exception as exc:
        raise ImportError("PyTorch is required to train GRU mood model. Install torch to continue.") from exc

    _set_seed(seed)
    torch.manual_seed(seed)

    if len(x) < 10:
        raise ValueError("Not enough sequence samples to train GRU model.")

    def _build_time_split_indices(meta_df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
        if meta_df.empty or "user_id" not in meta_df.columns:
            idx_all = np.arange(len(x))
            np.random.shuffle(idx_all)
            split_at = max(int(0.8 * len(idx_all)), 1)
            tr = idx_all[:split_at]
            va = idx_all[split_at:] if split_at < len(idx_all) else idx_all[:1]
            return tr.astype(int), va.astype(int)

        work = meta_df.copy().reset_index().rename(columns={"index": "row_idx"})
        if "target_created_at" in work.columns:
            work["target_created_at"] = pd.to_datetime(work["target_created_at"], errors="coerce", utc=True)
            work = work.sort_values(["user_id", "target_created_at", "target_idx", "row_idx"], kind="mergesort")
        else:
            work = work.sort_values(["user_id", "target_idx", "row_idx"], kind="mergesort")

        train_parts: list[np.ndarray] = []
        val_parts: list[np.ndarray] = []

        for _, g in work.groupby("user_id", sort=False):
            idxs = g["row_idx"].to_numpy(dtype=int)
            n = len(idxs)
            if n <= 2:
                train_parts.append(idxs)
                continue

            split_at = int(np.floor(0.8 * n))
            split_at = min(max(split_at, 1), n - 1)
            train_parts.append(idxs[:split_at])
            val_parts.append(idxs[split_at:])

        train_idx_out = np.concatenate(train_parts) if train_parts else np.array([], dtype=int)
        val_idx_out = np.concatenate(val_parts) if val_parts else np.array([], dtype=int)

        # Fallback guardrail if per-user split produces too little validation coverage.
        if len(train_idx_out) == 0 or len(val_idx_out) == 0:
            idx_all = np.arange(len(x))
            np.random.shuffle(idx_all)
            split_at = max(int(0.8 * len(idx_all)), 1)
            train_idx_out = idx_all[:split_at]
            val_idx_out = idx_all[split_at:] if split_at < len(idx_all) else idx_all[:1]
        return train_idx_out.astype(int), val_idx_out.astype(int)

    train_idx, val_idx = _build_time_split_indices(meta)

    x_train = torch.tensor(x[train_idx], dtype=torch.float32).unsqueeze(-1)
    y_train = torch.tensor(y[train_idx], dtype=torch.float32).unsqueeze(-1)
    x_val = torch.tensor(x[val_idx], dtype=torch.float32).unsqueeze(-1)
    y_val = torch.tensor(y[val_idx], dtype=torch.float32).unsqueeze(-1)

    # Naive baseline: predict next sentiment as last sentiment from the input window.
    val_baseline_pred = x[val_idx][:, -1]
    val_baseline_mse = float(np.mean((val_baseline_pred - y[val_idx]) ** 2)) if len(val_idx) else float("nan")

    train_loader = DataLoader(TensorDataset(x_train, y_train), batch_size=max(1, batch_size), shuffle=True)

    class MoodGRU(nn.Module):
        def __init__(self, hidden: int):
            super().__init__()
            self.gru = nn.GRU(input_size=1, hidden_size=hidden, num_layers=1, batch_first=True)
            self.head = nn.Linear(hidden, 1)

        def forward(self, seq):
            out, _ = self.gru(seq)
            pred = self.head(out[:, -1, :])
            return torch.tanh(pred)

    model = MoodGRU(hidden=max(8, int(hidden_size)))
    optimizer = torch.optim.Adam(model.parameters(), lr=float(learning_rate))
    criterion = nn.MSELoss()

    train_loss = 0.0
    for _ in range(max(1, int(epochs))):
        model.train()
        batch_losses: list[float] = []
        for xb, yb in train_loader:
            optimizer.zero_grad()
            pred = model(xb)
            loss = criterion(pred, yb)
            loss.backward()
            optimizer.step()
            batch_losses.append(float(loss.item()))
        train_loss = float(np.mean(batch_losses)) if batch_losses else 0.0

    model.eval()
    with torch.no_grad():
        val_pred = model(x_val)
        val_mse = float(criterion(val_pred, y_val).item())

    return model, train_loss, val_mse, val_baseline_mse, int(len(train_idx)), int(len(val_idx))


def _predict_sequences(model, x: np.ndarray) -> np.ndarray:
    import torch

    with torch.no_grad():
        pred = model(torch.tensor(x, dtype=torch.float32).unsqueeze(-1))
    return pred.detach().cpu().numpy().reshape(-1)


def _summarize_users(
    df: pd.DataFrame,
    meta: pd.DataFrame,
    x: np.ndarray,
    y: np.ndarray,
    pred: np.ndarray,
) -> pd.DataFrame:
    if len(pred) != len(meta):
        raise ValueError("Prediction/meta length mismatch.")

    pred_df = meta.copy()
    pred_df["target"] = y.astype(float)
    pred_df["pred_next_sentiment"] = pred.astype(float)
    pred_df["abs_err"] = (pred_df["target"] - pred_df["pred_next_sentiment"]).abs()

    # Actual sentiment volatility from raw timeline.
    raw = df.groupby("user_id", as_index=False).agg(messages=("sentiment_score", "size"))
    act = (
        df.sort_values(["user_id", "created_at", "sentiment_score"], kind="mergesort")
        .groupby("user_id", as_index=False)
        .apply(lambda g: pd.Series({"actual_volatility": float(g["sentiment_score"].diff().abs().mean() or 0.0)}))
        .reset_index(drop=True)
    )

    pred_agg = pred_df.groupby("user_id", as_index=False).agg(
        prediction_mae=("abs_err", "mean"),
        predicted_volatility=("pred_next_sentiment", lambda s: float(s.diff().abs().mean() or 0.0)),
    )

    trend_df = (
        df.groupby("user_id", as_index=False)
        .apply(
            lambda g: pd.Series(
                {
                    "recent_avg": float(g["sentiment_score"].tail(5).mean() if len(g) >= 1 else 0.0),
                    "prior_avg": float(g["sentiment_score"].iloc[-10:-5].mean() if len(g) >= 10 else g["sentiment_score"].head(5).mean()),
                }
            )
        )
        .reset_index(drop=True)
    )
    trend_df["trend_delta"] = trend_df["recent_avg"] - trend_df["prior_avg"]

    out = raw.merge(act, on="user_id", how="left").merge(pred_agg, on="user_id", how="inner").merge(
        trend_df[["user_id", "trend_delta"]],
        on="user_id",
        how="left",
    )

    out["actual_volatility"] = pd.to_numeric(out["actual_volatility"], errors="coerce").fillna(0.0)
    out["predicted_volatility"] = pd.to_numeric(out["predicted_volatility"], errors="coerce").fillna(0.0)
    out["prediction_mae"] = pd.to_numeric(out["prediction_mae"], errors="coerce").fillna(0.0)

    av = out["actual_volatility"]
    pv = out["predicted_volatility"]
    av_norm = av / av.max() if av.max() > 0 else av
    pv_norm = pv / pv.max() if pv.max() > 0 else pv
    out["mood_swing_index"] = (0.65 * av_norm + 0.35 * pv_norm).clip(0.0, 1.0)

    q80 = float(out["mood_swing_index"].quantile(0.80)) if not out.empty else 0.0
    q60 = float(out["mood_swing_index"].quantile(0.60)) if not out.empty else 0.0

    def risk(v: float) -> str:
        if v >= q80:
            return "High"
        if v >= q60:
            return "Medium"
        return "Low"

    def trend_tag(v: float) -> str:
        if v > 0.03:
            return "Improving"
        if v < -0.03:
            return "Worsening"
        return "Stable"

    def recommendation(row: pd.Series) -> str:
        if row["risk_flag"] == "High":
            return "Monitor recent conversation context and collect more timely feedback labels."
        if row["risk_flag"] == "Medium":
            return "Add recent labeled samples and review time-of-day interaction patterns."
        return "Continue periodic monitoring and retrain weekly to catch drift."

    out["risk_flag"] = out["mood_swing_index"].apply(risk)
    out["trend"] = out["trend_delta"].apply(trend_tag)
    out["recommendation"] = out.apply(recommendation, axis=1)
    out = out.sort_values("mood_swing_index", ascending=False).reset_index(drop=True)
    return out[
        [
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
    ]


def main() -> None:
    args = parse_args()
    sentiment_path = Path(args.sentiment)
    sessions_path = resolve_input_path(args.sessions)
    out_summary = Path(args.out_summary)
    out_report = Path(args.out_report)

    df = _load_sentiment_messages(sentiment_path, sessions_path)
    x, y, meta = _build_samples(
        df=df,
        sequence_length=int(args.sequence_length),
        min_user_msgs=int(args.min_user_msgs),
    )
    if len(x) == 0:
        warning = (
            "Skipping GRU mood swing training because no eligible user sequences were found. "
            "Seed/demo data usually needs lower message volume or more real sentiment history."
        )
        write_empty_outputs(
            out_summary=out_summary,
            out_report=out_report,
            total_messages=int(len(df)),
            eligible_users=0,
            args=args,
            warning=warning,
        )
        return

    try:
        model, train_loss, val_mse, val_baseline_mse, train_samples, val_samples = _train_gru(
            x=x,
            y=y,
            meta=meta,
            hidden_size=int(args.hidden_size),
            epochs=int(args.epochs),
            batch_size=int(args.batch_size),
            learning_rate=float(args.learning_rate),
            seed=int(args.seed),
        )
    except (ValueError, ImportError) as exc:
        warning = f"Skipping GRU mood swing training because the dataset is too small or runtime requirements are missing. Details: {exc}"
        write_empty_outputs(
            out_summary=out_summary,
            out_report=out_report,
            total_messages=int(len(df)),
            eligible_users=int(meta["eligible_users"].iloc[0]) if "eligible_users" in meta.columns and not meta.empty else 0,
            args=args,
            warning=warning,
        )
        return
    pred = _predict_sequences(model, x)
    summary = _summarize_users(df=df, meta=meta, x=x, y=y, pred=pred)

    report = pd.DataFrame(
        [
            {
                "total_messages": int(len(df)),
                "eligible_users": int(meta["eligible_users"].iloc[0]) if "eligible_users" in meta.columns and not meta.empty else 0,
                "sequence_length": int(args.sequence_length),
                "hidden_size": int(args.hidden_size),
                "epochs": int(args.epochs),
                "batch_size": int(args.batch_size),
                "train_samples": int(train_samples),
                "val_samples": int(val_samples),
                "train_loss": float(train_loss),
                "val_mse": float(val_mse),
                "val_baseline_mse": float(val_baseline_mse),
                "val_mse_vs_baseline_pct": float(((val_baseline_mse - val_mse) / val_baseline_mse) * 100.0)
                if pd.notna(val_baseline_mse) and val_baseline_mse > 0
                else np.nan,
                "validation_split_strategy": "per_user_chronological_80_20",
            }
        ]
    )

    save_artifact_df(summary, "gru_mood_swing_summary", out_summary, index=False)
    save_artifact_df(report, "gru_mood_training_report", out_report, index=False)

    print(f"[ok] Saved mood swing summary: {out_summary}")
    print(f"[ok] Saved training report: {out_report}")
    print(
        f"[ok] Users scored: {len(summary)} | train_loss={train_loss:.6f} | val_mse={val_mse:.6f} "
        f"| val_baseline_mse={val_baseline_mse:.6f}"
    )


if __name__ == "__main__":
    main()
