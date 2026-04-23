"""
Train an XGBoost classifier on GraphSAGE user embeddings to predict sentiment_label
(Positive vs Negative), then explain feature importance with SHAP.

Inputs:
  - user_embeddings.csv (must include user_id and emb_* columns)
  - sentiment_features.csv OR sentiment_scores.csv
    * If sentiment file lacks user_id but has session_id, pass sessions.csv to map session_id -> user_id.

Outputs:
  - shap_summary.png
  - xgb_embedding_feature_importance.csv

Usage:
  /path/to/python train_xgb_shap_sentiment.py \
      --embeddings user_embeddings.csv \
      --sentiment sentiment_features.csv \
      --out_plot shap_summary.png
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Optional

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold, train_test_split
from xgboost import XGBClassifier

from app_config import EMBEDDINGS_ARTIFACT_DIR, GNN_PREPROCESSED_DIR, SECRET_DATA_DIR, SENTIMENT_ARTIFACT_DIR, XGB_ARTIFACT_DIR
from lib.device_utils import resolve_xgb_device
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


def write_placeholder_plot(plot_path: Path, message: str) -> None:
    plot_path.parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(10, 3))
    plt.axis("off")
    plt.text(0.5, 0.5, message, ha="center", va="center", wrap=True)
    plt.tight_layout()
    plt.savefig(plot_path, dpi=180)
    plt.close()


def write_empty_xgb_outputs(
    args: argparse.Namespace,
    emb: pd.DataFrame,
    target_meta: dict,
    warning: str,
) -> None:
    out_importance = Path(args.out_importance)
    out_predictions = Path(args.out_predictions)
    out_target_report = Path(args.out_target_report)
    out_model = Path(args.out_model)
    merged_path = out_importance.with_name("xgb_embedding_feature_importance_merged.csv")

    empty_importance = pd.DataFrame(columns=["feature", "mean_abs_shap", "feature_label"])
    empty_merged = pd.DataFrame(columns=["signal_name", "mean_abs_shap", "dim_count", "dimensions"])
    empty_predictions = pd.DataFrame(
        columns=[
            "user_id",
            "target",
            "pred_label",
            "pred_prob_positive",
            "predicted_class",
            "confidence",
        ]
    )
    if "user_id" in emb.columns and not emb.empty:
        empty_predictions = pd.DataFrame(
            {
                "user_id": pd.to_numeric(emb["user_id"], errors="coerce").dropna().astype(int),
                "target": np.nan,
                "pred_label": np.nan,
                "pred_prob_positive": np.nan,
                "predicted_class": "unavailable",
                "confidence": np.nan,
            }
        )

    report = {
        "target_source": target_meta.get("target_source", ""),
        "human_label_column": target_meta.get("human_label_column", ""),
        "human_label_rows": target_meta.get("human_label_rows", 0),
        "pseudo_label_rows": target_meta.get("pseudo_label_rows", 0),
        "joined_users": 0,
        "train_rows": 0,
        "test_rows": 0,
        "train_neg": 0,
        "train_pos": 0,
        "train_original_neg": 0,
        "train_original_pos": 0,
        "train_rebalance_applied": False,
        "scale_pos_weight": np.nan,
        "cv_folds_requested": int(args.cv_folds),
        "cv_folds_used": 0,
        "early_stopping_rounds": int(args.early_stopping_rounds),
        "best_iteration": np.nan,
        "cv_auc_mean": np.nan,
        "cv_auc_std": np.nan,
        "cv_best_iter_mean": np.nan,
        "cv_best_iter_std": np.nan,
        "accuracy": np.nan,
        "auc": np.nan,
        "pred_majority_share": np.nan,
        "pred_prob_std": np.nan,
        "model_collapse_risk": False,
        "model_artifact": "",
        "warning": warning,
        "original_dims": 0,
        "kept_dims": 0,
        "dropped_low_variance": 0,
        "dropped_high_corr": 0,
        "winsor_q_low": float(args.winsor_q_low),
        "winsor_q_high": float(args.winsor_q_high),
        "variance_threshold": float(args.low_variance_threshold),
        "corr_drop_threshold": float(args.corr_drop_threshold),
    }

    save_artifact_df(empty_importance, "xgb_embedding_feature_importance", out_importance, index=False)
    save_artifact_df(empty_merged, "xgb_embedding_feature_importance_merged", merged_path, index=False)
    save_artifact_df(empty_predictions, "xgb_user_predictions", out_predictions, index=False)
    save_artifact_df(pd.DataFrame([report]), "xgb_target_report", out_target_report, index=False)
    write_placeholder_plot(Path(args.out_plot), warning)

    if out_model.exists():
        out_model.unlink()

    print(f"[warning] {warning}")
    print(f"[saved] Placeholder SHAP summary plot: {args.out_plot}")
    print(f"[saved] Empty feature importance CSV: {args.out_importance}")
    print(f"[saved] Empty merged feature importance CSV: {merged_path}")
    print(f"[saved] Target report CSV: {args.out_target_report}")
    print(f"[saved] User predictions CSV: {args.out_predictions}")


def parse_args() -> argparse.Namespace:
    base = XGB_ARTIFACT_DIR
    p = argparse.ArgumentParser(description="Train XGBoost + SHAP on user embeddings for sentiment prediction")
    p.add_argument("--embeddings", type=str, default=str(EMBEDDINGS_ARTIFACT_DIR / "user_embeddings.csv"))
    p.add_argument("--sentiment", type=str, default=str(SENTIMENT_ARTIFACT_DIR / "sentiment_scores.csv"))
    p.add_argument("--sessions", type=str, default=str(SECRET_DATA_DIR / "sessions.csv"))
    p.add_argument(
        "--target_source",
        type=str,
        choices=["auto", "pseudo"],
        default="auto",
        help="target labels are derived from sentiment artifacts",
    )
    p.add_argument(
        "--allow_pseudo_fallback",
        action="store_true",
        help="allow fallback to pseudo sentiment labels when human labels are unavailable",
    )
    p.add_argument("--out_plot", type=str, default=str(base / "shap_summary.png"))
    p.add_argument("--out_importance", type=str, default=str(base / "xgb_embedding_feature_importance.csv"))
    p.add_argument("--out_target_report", type=str, default=str(base / "xgb_target_report.csv"))
    p.add_argument("--out_predictions", type=str, default=str(base / "xgb_user_predictions.csv"))
    p.add_argument("--out_model", type=str, default=str(base / "xgb_model.json"))
    p.add_argument("--winsor_q_low", type=float, default=0.01)
    p.add_argument("--winsor_q_high", type=float, default=0.99)
    p.add_argument("--low_variance_threshold", type=float, default=1e-6)
    p.add_argument("--corr_drop_threshold", type=float, default=0.995)
    p.add_argument("--test_size", type=float, default=0.25)
    p.add_argument("--eval_size", type=float, default=0.20, help="Validation split fraction inside training data")
    p.add_argument("--cv_folds", type=int, default=5, help="Requested StratifiedKFold splits for CV")
    p.add_argument("--early_stopping_rounds", type=int, default=40)
    p.add_argument("--n_estimators", type=int, default=1200)
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def pick_sentiment_file(path_arg: str) -> Path:
    p = Path(path_arg)
    if p.exists():
        return p
    fallback = p.parent / "sentiment_scores.csv"
    if fallback.exists():
        print(f"[info] Sentiment file not found at {p}, using fallback {fallback}")
        return fallback
    gnn_fallback = GNN_PREPROCESSED_DIR / "messages_nodes.csv"
    if gnn_fallback.exists():
        print(f"[info] Sentiment file not found at {p}, using fallback {gnn_fallback}")
        return gnn_fallback
    raise FileNotFoundError(f"Could not find sentiment file: {p}")


def normalize_label(lbl: str) -> str:
    s = str(lbl).strip().lower()
    if s in {"positive", "pos", "1", "true"}:
        return "positive"
    if s in {"negative", "neg", "0", "false"}:
        return "negative"
    return "other"


def prettify_embedding_feature_name(feature: str) -> str:
    s = str(feature).strip()
    if s.startswith("emb_"):
        idx = s.split("_", 1)[1]
        return f"Embedding Dimension {idx}"
    return s


def load_embedding_label_map() -> dict[str, str]:
    labels_path = EMBEDDINGS_ARTIFACT_DIR / "embedding_dimension_labels.csv"
    if not labels_path.exists():
        return {}

    lbl = load_artifact_df("embedding_dimension_labels", labels_path)
    if lbl.empty or "feature" not in lbl.columns:
        return {}

    label_col = "label" if "label" in lbl.columns else None
    if label_col is None:
        return {}

    out: dict[str, str] = {}
    for _, row in lbl.iterrows():
        feat = str(row.get("feature", "")).strip()
        raw_lbl = str(row.get(label_col, "")).strip()
        if not feat:
            continue
        out[feat] = raw_lbl if raw_lbl else prettify_embedding_feature_name(feat)
    return out


def preprocess_embedding_matrix(
    X: pd.DataFrame,
    winsor_q_low: float = 0.01,
    winsor_q_high: float = 0.99,
    low_variance_threshold: float = 1e-6,
    corr_drop_threshold: float = 0.995,
) -> tuple[pd.DataFrame, dict]:
    work = X.copy()
    work = work.apply(pd.to_numeric, errors="coerce").fillna(0.0).astype(np.float32)
    original_cols = list(work.columns)

    q_low = work.quantile(float(winsor_q_low))
    q_high = work.quantile(float(winsor_q_high))
    work = work.clip(lower=q_low, upper=q_high, axis=1)

    var = work.var(axis=0)
    keep_var_cols = var[var > float(low_variance_threshold)].index.tolist()
    dropped_low_var = [c for c in work.columns if c not in keep_var_cols]
    work = work[keep_var_cols].copy()

    dropped_high_corr: list[str] = []
    if work.shape[1] > 1 and float(corr_drop_threshold) < 1.0:
        corr = work.corr().abs()
        upper = corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool))
        for col in upper.columns:
            if (upper[col] > float(corr_drop_threshold)).any():
                dropped_high_corr.append(col)
        if dropped_high_corr:
            work = work.drop(columns=sorted(set(dropped_high_corr)), errors="ignore")

    stats = {
        "original_dims": int(len(original_cols)),
        "kept_dims": int(work.shape[1]),
        "dropped_low_variance": int(len(dropped_low_var)),
        "dropped_high_corr": int(len(set(dropped_high_corr))),
        "winsor_q_low": float(winsor_q_low),
        "winsor_q_high": float(winsor_q_high),
        "variance_threshold": float(low_variance_threshold),
        "corr_drop_threshold": float(corr_drop_threshold),
    }
    return work, stats


def rebalance_binary_train_data(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    seed: int,
) -> tuple[pd.DataFrame, pd.Series, dict[str, int | bool]]:
    X_in = X_train.reset_index(drop=True).copy()
    y_in = pd.Series(y_train).reset_index(drop=True).astype(int)

    meta: dict[str, int | bool] = {
        "applied": False,
        "original_rows": int(len(y_in)),
        "original_neg": int((y_in == 0).sum()),
        "original_pos": int((y_in == 1).sum()),
        "balanced_rows": int(len(y_in)),
        "balanced_neg": int((y_in == 0).sum()),
        "balanced_pos": int((y_in == 1).sum()),
    }

    if y_in.nunique() < 2:
        return X_in, y_in, meta

    counts = y_in.value_counts()
    if len(counts) < 2 or int(counts.iloc[0]) == int(counts.iloc[1]):
        return X_in, y_in, meta

    majority_class = int(counts.idxmax())
    minority_class = int(counts.idxmin())
    majority_count = int(counts.max())
    minority_count = int(counts.min())

    target_per_class = int(max(2, round((majority_count + minority_count) / 2)))

    train_df = X_in.copy()
    train_df["_target"] = y_in.values
    major_df = train_df[train_df["_target"] == majority_class]
    minor_df = train_df[train_df["_target"] == minority_class]

    major_bal = major_df.sample(
        n=target_per_class,
        replace=(len(major_df) < target_per_class),
        random_state=seed,
    )
    minor_bal = minor_df.sample(
        n=target_per_class,
        replace=(len(minor_df) < target_per_class),
        random_state=seed + 1,
    )

    balanced_df = (
        pd.concat([major_bal, minor_bal], axis=0, ignore_index=True)
        .sample(frac=1.0, random_state=seed)
        .reset_index(drop=True)
    )
    y_bal = balanced_df.pop("_target").astype(int)
    X_bal = balanced_df.copy()

    meta.update(
        {
            "applied": True,
            "balanced_rows": int(len(y_bal)),
            "balanced_neg": int((y_bal == 0).sum()),
            "balanced_pos": int((y_bal == 1).sum()),
        }
    )
    return X_bal, y_bal, meta


def _normalize_binary_feedback(v) -> Optional[int]:
    if pd.isna(v):
        return None

    s = str(v).strip().lower()
    if s in {"positive", "pos", "good", "great", "up", "thumbs_up", "like", "liked", "satisfied", "true", "1"}:
        return 1
    if s in {"negative", "neg", "bad", "poor", "down", "thumbs_down", "dislike", "dissatisfied", "false", "0"}:
        return 0

    try:
        n = float(s)
        # Typical 1..5 style ratings.
        if n >= 4:
            return 1
        if n <= 2:
            return 0
    except Exception:
        return None

    return None


def _heuristic_polarity(text: str) -> float:
    s = str(text or "").strip().lower()
    if not s:
        return 0.0
    neg_terms = {
        "bad", "worse", "worst", "hate", "angry", "upset", "frustrated", "annoyed", "terrible", "awful",
        "slow", "broken", "error", "issue", "problem", "failed", "failure", "crash", "bug",
    }
    pos_terms = {
        "good", "great", "awesome", "nice", "love", "happy", "thanks", "resolved", "perfect", "excellent",
        "fast", "smooth", "helpful", "fixed", "works", "working",
    }
    tokens = [t for t in str(s).replace("!", " ").replace("?", " ").split() if t]
    if not tokens:
        return 0.0
    pos = sum(1 for t in tokens if t in pos_terms)
    neg = sum(1 for t in tokens if t in neg_terms)
    return float((pos - neg) / max(len(tokens), 6))


def prepare_human_feedback_labels(feedback: pd.DataFrame) -> tuple[pd.DataFrame, Optional[str], int]:
    f = feedback.copy()
    if "user_id" not in f.columns:
        return pd.DataFrame(columns=["user_id", "target", "target_label"]), None, 0

    f["user_id"] = pd.to_numeric(f["user_id"], errors="coerce")
    f = f.dropna(subset=["user_id"]).copy()
    f["user_id"] = f["user_id"].astype(int)
    if f.empty:
        return pd.DataFrame(columns=["user_id", "target", "target_label"]), None, 0

    candidate_cols = [
        "sentiment_label",
        "label",
        "feedback_label",
        "thumb",
        "thumbs",
        "is_positive",
        "rating",
        "score",
    ]
    available = [c for c in candidate_cols if c in f.columns]
    if not available:
        return pd.DataFrame(columns=["user_id", "target", "target_label"]), None, 0

    best_col = None
    best_valid = -1
    best_series = None
    for c in available:
        mapped = f[c].apply(_normalize_binary_feedback)
        valid = int(mapped.notna().sum())
        if valid > best_valid:
            best_valid = valid
            best_col = c
            best_series = mapped

    if best_col is None or best_series is None or best_valid <= 0:
        return pd.DataFrame(columns=["user_id", "target", "target_label"]), None, 0

    work = f[["user_id"]].copy()
    work["target"] = best_series
    work = work.dropna(subset=["target"]).copy()
    work["target"] = work["target"].astype(int)
    if work.empty:
        return pd.DataFrame(columns=["user_id", "target", "target_label"]), best_col, 0

    agg = (
        work.groupby(["user_id", "target"]).size().reset_index(name="cnt")
        .sort_values(["user_id", "cnt"], ascending=[True, False])
    )
    user_label = agg.drop_duplicates("user_id").copy()
    user_label["target_label"] = np.where(user_label["target"] == 1, "positive", "negative")
    return user_label[["user_id", "target", "target_label"]], best_col, int(len(work))


def prepare_pseudo_sentiment_labels(sentiment: pd.DataFrame, sessions_path: Path) -> pd.DataFrame:
    s = sentiment.copy()

    if "sentiment_label" not in s.columns:
        raise ValueError("Sentiment file must include sentiment_label column")

    if "user_id" not in s.columns:
        if "session_id" not in s.columns:
            raise ValueError("Sentiment file needs either user_id or session_id for joining")
        if not sessions_path.exists():
            raise FileNotFoundError(
                f"Sentiment file has no user_id; sessions mapping file not found: {sessions_path}"
            )
        sessions = pd.read_csv(sessions_path, usecols=["id", "user_id"])
        sessions["id"] = pd.to_numeric(sessions["id"], errors="coerce")
        sessions["user_id"] = pd.to_numeric(sessions["user_id"], errors="coerce")

        s["session_id"] = pd.to_numeric(s["session_id"], errors="coerce")
        s = s.merge(sessions.rename(columns={"id": "session_id"}), on="session_id", how="left")

    s["user_id"] = pd.to_numeric(s["user_id"], errors="coerce")
    s = s.dropna(subset=["user_id"]).copy()
    s["user_id"] = s["user_id"].astype(int)

    s["sentiment_label_norm"] = s["sentiment_label"].apply(normalize_label)
    s = s[s["sentiment_label_norm"].isin(["positive", "negative"])].copy()
    if s.empty:
        # Recovery path: derive binary labels from sentiment_score if labels are mostly neutral.
        if "sentiment_score" not in sentiment.columns:
            raise ValueError("No Positive/Negative rows left after filtering sentiment_label")

        ss = sentiment.copy()
        if "user_id" not in ss.columns:
            if "session_id" not in ss.columns:
                raise ValueError("Sentiment file needs either user_id or session_id for joining")
            if not sessions_path.exists():
                raise FileNotFoundError(
                    f"Sentiment file has no user_id; sessions mapping file not found: {sessions_path}"
                )
            sessions = pd.read_csv(sessions_path, usecols=["id", "user_id"])
            sessions["id"] = pd.to_numeric(sessions["id"], errors="coerce")
            sessions["user_id"] = pd.to_numeric(sessions["user_id"], errors="coerce")
            ss["session_id"] = pd.to_numeric(ss["session_id"], errors="coerce")
            ss = ss.merge(sessions.rename(columns={"id": "session_id"}), on="session_id", how="left")

        ss["user_id"] = pd.to_numeric(ss["user_id"], errors="coerce")
        ss["sentiment_score"] = pd.to_numeric(ss["sentiment_score"], errors="coerce")
        ss = ss.dropna(subset=["user_id", "sentiment_score"]).copy()
        ss = ss[(ss["sentiment_score"] >= 0.05) | (ss["sentiment_score"] <= -0.05)].copy()
        if ss.empty and "message" in sentiment.columns:
            ss = sentiment.copy()
            if "user_id" not in ss.columns and "session_id" in ss.columns and sessions_path.exists():
                sessions = pd.read_csv(sessions_path, usecols=["id", "user_id"])
                sessions["id"] = pd.to_numeric(sessions["id"], errors="coerce")
                sessions["user_id"] = pd.to_numeric(sessions["user_id"], errors="coerce")
                ss["session_id"] = pd.to_numeric(ss["session_id"], errors="coerce")
                ss = ss.merge(sessions.rename(columns={"id": "session_id"}), on="session_id", how="left")
            ss["user_id"] = pd.to_numeric(ss["user_id"], errors="coerce")
            ss["sentiment_score"] = ss["message"].apply(_heuristic_polarity)
            ss = ss.dropna(subset=["user_id", "sentiment_score"]).copy()
            ss = ss[(ss["sentiment_score"] >= 0.02) | (ss["sentiment_score"] <= -0.02)].copy()
        if ss.empty:
            raise ValueError("No Positive/Negative rows left after filtering sentiment_label")
        ss["user_id"] = ss["user_id"].astype(int)
        ss["sentiment_label_norm"] = np.where(ss["sentiment_score"] > 0, "positive", "negative")
        s = ss

    # Aggregate message-level labels to user-level target via majority vote.
    agg = (
        s.groupby(["user_id", "sentiment_label_norm"]).size().reset_index(name="cnt")
        .sort_values(["user_id", "cnt"], ascending=[True, False])
    )
    user_label = agg.drop_duplicates("user_id").copy()
    user_label["target"] = (user_label["sentiment_label_norm"] == "positive").astype(int)
    return user_label[["user_id", "sentiment_label_norm", "target"]]


def resolve_targets(args: argparse.Namespace, sentiment: pd.DataFrame, sessions_path: Path) -> tuple[pd.DataFrame, dict]:
    source_mode = args.target_source

    meta = {
        "target_source": "",
        "human_label_column": "",
        "human_label_rows": 0,
        "pseudo_label_rows": 0,
        "warning": "",
    }

    if source_mode == "auto" and not args.allow_pseudo_fallback:
        raise ValueError(
            "Pseudo-label fallback is disabled. Re-run with --allow_pseudo_fallback."
        )

    pseudo = prepare_pseudo_sentiment_labels(sentiment, sessions_path=sessions_path)
    meta["target_source"] = "pseudo_sentiment_label"
    meta["pseudo_label_rows"] = int(len(pseudo))
    meta["warning"] = (
        "Pseudo-label target in use (likely noisy). SHAP explains proxy sentiment, not validated human sentiment."
    )
    return pseudo[["user_id", "target"]].copy(), meta


def main() -> None:
    args = parse_args()

    # Resolve hardware-accelerated device once for this run
    xgb_device = resolve_xgb_device()

    emb_path = Path(args.embeddings)
    sent_path = pick_sentiment_file(args.sentiment)
    sessions_path = resolve_input_path(args.sessions)

    emb = load_artifact_df("user_embeddings", emb_path)
    if "user_id" not in emb.columns:
        raise ValueError("Embeddings file must contain user_id column")

    emb["user_id"] = pd.to_numeric(emb["user_id"], errors="coerce")
    emb = emb.dropna(subset=["user_id"]).copy()
    emb["user_id"] = emb["user_id"].astype(int)

    emb_cols = [c for c in emb.columns if c.startswith("emb_")]
    if not emb_cols:
        raise ValueError("No embedding columns found. Expected columns like emb_0 ... emb_63")
    emb_label_map = load_embedding_label_map()

    emb[emb_cols] = emb[emb_cols].apply(pd.to_numeric, errors="coerce").fillna(0.0).astype(np.float32)

    sentiment = pd.read_csv(sent_path)
    try:
        user_targets, target_meta = resolve_targets(args, sentiment, sessions_path=sessions_path)
    except ValueError as exc:
        warning = (
            "Skipping XGBoost sentiment training because no usable binary sentiment targets were found. "
            f"Details: {exc}"
        )
        write_empty_xgb_outputs(args, emb, {"warning": warning}, warning)
        return

    data = emb.merge(user_targets, on="user_id", how="inner")
    if data.empty:
        warning = (
            "Skipping XGBoost sentiment training because embeddings and resolved sentiment targets did not overlap "
            "on any user_id values."
        )
        write_empty_xgb_outputs(args, emb, target_meta, warning)
        return

    X_raw = data[emb_cols].copy()
    X, prep_stats = preprocess_embedding_matrix(
        X_raw,
        winsor_q_low=args.winsor_q_low,
        winsor_q_high=args.winsor_q_high,
        low_variance_threshold=args.low_variance_threshold,
        corr_drop_threshold=args.corr_drop_threshold,
    )
    emb_cols_used = X.columns.tolist()
    if not emb_cols_used:
        raise ValueError("All embedding dimensions were removed by preprocessing. Relax preprocessing thresholds.")
    y = data["target"].astype(int).copy()
    if len(data) < 2 or y.nunique() < 2:
        warning = (
            "Skipping XGBoost sentiment training because the resolved target set is too small or contains only one "
            "class after joining with embeddings."
        )
        write_empty_xgb_outputs(args, emb, target_meta, warning)
        return

    # Handle small datasets gracefully for dev/test environments.
    # stratified split requires at least 2 samples per class in the test set.
    can_stratify = (y.nunique() > 1) and (y.value_counts().min() >= 2)
    strat = y if can_stratify else None
    
    # Ensure test_size results in at least 1 sample, but if N is tiny, don't split at all for XGB.
    if len(X) < 5:
        print("[warning] Dataset too small for reliable train/test split. Using all data for training (SHAP summary mode).")
        X_train, X_test, y_train, y_test = X, X, y, y
    else:
        actual_test_size = max(1, int(len(X) * args.test_size))
        if actual_test_size < 2 and can_stratify:
            print("[info] Test size too small for stratified split; disabling stratification.")
            strat = None
            
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=args.test_size, random_state=args.seed, stratify=strat
        )

    X_train_model, y_train_model, balance_meta = rebalance_binary_train_data(X_train, y_train, seed=args.seed)
    # ... rest of rebalance print ...
    
    neg = int((y_train_model == 0).sum())
    pos = int((y_train_model == 1).sum())
    # Only use scale_pos_weight when positive class is the minority.
    scale_pos_weight = (neg / pos) if (pos > 0 and pos < neg) else 1.0
    min_class_count = int(min(neg, pos))
    requested_cv_folds = max(2, int(args.cv_folds))
    cv_folds_used = int(min(requested_cv_folds, min_class_count)) if min_class_count >= 2 else 0
    
    # Disable CV if samples are too few
    if len(y_train_model) < 5 or cv_folds_used < 2:
        print("[info] Skipping CV AUC due to insufficient samples in training split.")
        cv_folds_used = 0

    class_weight_map: dict[int, float] = {0: 1.0, 1: 1.0}
    if min_class_count > 0 and neg != pos:
        if neg < pos:
            class_weight_map[0] = float(pos / neg)
        else:
            class_weight_map[1] = float(neg / pos)
    sample_weight_train = y_train_model.map(class_weight_map).astype(float)

    class_balance_warning = ""
    if min_class_count < 10:
        class_balance_warning = (
            f"Very small minority class in training split (neg={neg}, pos={pos}). "
            "Metrics may be unstable; prioritize evaluation on human-labeled holdout data."
        )
        print(f"[warning] {class_balance_warning}")

    model_params = dict(
        n_estimators=int(args.n_estimators),
        max_depth=5,
        learning_rate=0.05,
        subsample=0.9,
        colsample_bytree=0.9,
        objective="binary:logistic",
        eval_metric="logloss",
        early_stopping_rounds=int(args.early_stopping_rounds),
        scale_pos_weight=scale_pos_weight,
        random_state=args.seed,
        device=xgb_device,
    )
    model_params_no_es = {k: v for k, v in model_params.items() if k != "early_stopping_rounds"}

    cv_auc_mean = float("nan")
    cv_auc_std = float("nan")
    cv_best_iter_mean = float("nan")
    cv_best_iter_std = float("nan")
    if cv_folds_used >= 2 and y_train_model.nunique() > 1:
        if cv_folds_used < requested_cv_folds:
            print(
                f"[warning] Requested {requested_cv_folds}-fold CV reduced to {cv_folds_used} due to minority class size."
            )
        cv = StratifiedKFold(n_splits=cv_folds_used, shuffle=True, random_state=args.seed)
        fold_aucs: list[float] = []
        fold_best_iters: list[float] = []
        for fold_idx, (tr_idx, va_idx) in enumerate(cv.split(X_train_model, y_train_model), start=1):
            X_tr_fold = X_train_model.iloc[tr_idx]
            y_tr_fold = y_train_model.iloc[tr_idx]
            X_va_fold = X_train_model.iloc[va_idx]
            y_va_fold = y_train_model.iloc[va_idx]

            model_fold = XGBClassifier(**model_params)
            sample_weight_fold = y_tr_fold.map(class_weight_map).astype(float)
            model_fold.fit(
                X_tr_fold,
                y_tr_fold,
                sample_weight=sample_weight_fold,
                eval_set=[(X_va_fold, y_va_fold)],
                verbose=False,
            )

            proba_fold = model_fold.predict_proba(X_va_fold)[:, 1]
            auc_fold = roc_auc_score(y_va_fold, proba_fold) if y_va_fold.nunique() > 1 else float("nan")
            if not np.isnan(auc_fold):
                fold_aucs.append(float(auc_fold))

            best_iter = getattr(model_fold, "best_iteration", None)
            if best_iter is not None:
                fold_best_iters.append(float(best_iter))

            print(
                f"[cv] fold={fold_idx}/{cv_folds_used} auc={auc_fold if np.isnan(auc_fold) else round(auc_fold, 4)} "
                f"best_iter={best_iter if best_iter is not None else 'n/a'}"
            )

        if fold_aucs:
            cv_auc_mean = float(np.mean(fold_aucs))
            cv_auc_std = float(np.std(fold_aucs))
        if fold_best_iters:
            cv_best_iter_mean = float(np.mean(fold_best_iters))
            cv_best_iter_std = float(np.std(fold_best_iters))

        print(
            f"[metrics] cv_auc_mean={cv_auc_mean if np.isnan(cv_auc_mean) else round(cv_auc_mean, 4)} "
            f"cv_auc_std={cv_auc_std if np.isnan(cv_auc_std) else round(cv_auc_std, 4)} "
            f"(stratified_{cv_folds_used}fold, early_stopping={args.early_stopping_rounds})"
        )
    else:
        print("[info] Skipping CV AUC due to insufficient minority samples in training split.")

    # Final fit with an internal validation split and early stopping.
    model = XGBClassifier(**model_params)
    best_iteration_final = float("nan")
    try:
        train_strat = y_train_model if y_train_model.nunique() > 1 else None
        X_fit, X_eval, y_fit, y_eval = train_test_split(
            X_train_model,
            y_train_model,
            test_size=float(args.eval_size),
            random_state=args.seed,
            stratify=train_strat,
        )
        if y_eval.nunique() > 1 and len(X_eval) > 0:
            sample_weight_fit = y_fit.map(class_weight_map).astype(float)
            model.fit(
                X_fit,
                y_fit,
                sample_weight=sample_weight_fit,
                eval_set=[(X_eval, y_eval)],
                verbose=False,
            )
            bi = getattr(model, "best_iteration", None)
            if bi is not None:
                best_iteration_final = float(bi)
            print(
                f"[fit] early_stopping_rounds={args.early_stopping_rounds} "
                f"best_iteration={bi if bi is not None else 'n/a'}"
            )
        else:
            model = XGBClassifier(**model_params_no_es)
            model.fit(X_train_model, y_train_model, sample_weight=sample_weight_train, verbose=False)
            print("[fit] early stopping skipped: validation split had a single class.")
    except Exception as exc:
        model = XGBClassifier(**model_params_no_es)
        model.fit(X_train_model, y_train_model, sample_weight=sample_weight_train, verbose=False)
        print(f"[warning] early stopping disabled due to split/fit issue: {exc}")

    # Persist the trained model so the dashboard can load/verify run artifacts.
    model_out_path = Path(args.out_model)
    model_out_path.parent.mkdir(parents=True, exist_ok=True)
    model.save_model(str(model_out_path))
    print(f"[saved] Model artifact: {model_out_path}")

    pred = model.predict(X_test)
    proba = model.predict_proba(X_test)[:, 1]
    acc = accuracy_score(y_test, pred)
    auc = roc_auc_score(y_test, proba) if y_test.nunique() > 1 else float("nan")
    print(f"[metrics] accuracy={acc:.4f} auc={auc if np.isnan(auc) else round(auc, 4)}")
    print(f"[data] joined_users={len(data)} train={len(X_train_model)} test={len(X_test)}")
    print(f"[target] source={target_meta.get('target_source','unknown')}")
    if target_meta.get("human_label_column"):
        print(f"[target] human_label_column={target_meta['human_label_column']}")
    if target_meta.get("warning"):
        print(f"[warning] {target_meta['warning']}")

    explainer = shap.TreeExplainer(model)
    shap_vals = explainer.shap_values(X)
    if isinstance(shap_vals, list):
        shap_arr = np.array(shap_vals[1]) if len(shap_vals) > 1 else np.array(shap_vals[0])
    else:
        shap_arr = np.array(shap_vals)

    mean_abs_shap = np.abs(shap_arr).mean(axis=0)
    imp = pd.DataFrame({"feature": emb_cols_used, "mean_abs_shap": mean_abs_shap}).sort_values(
        "mean_abs_shap", ascending=False
    )
    imp["feature_label"] = imp["feature"].map(emb_label_map).fillna(imp["feature"].map(prettify_embedding_feature_name))
    save_artifact_df(imp, "xgb_embedding_feature_importance", Path(args.out_importance), index=False)

    # Save per-user predictions so dashboard can show concrete model outputs.
    proba_all = model.predict_proba(X)[:, 1]
    pred_all = (proba_all >= 0.5).astype(int)
    pred_df = pd.DataFrame(
        {
            "user_id": data["user_id"].astype(int),
            "target": y.astype(int),
            "pred_label": pred_all.astype(int),
            "pred_prob_positive": proba_all.astype(float),
            "predicted_class": np.where(pred_all == 1, "positive", "negative"),
            "confidence": (2.0 * np.abs(proba_all - 0.5)).astype(float),
        }
    ).sort_values("pred_prob_positive", ascending=False)
    save_artifact_df(pred_df, "xgb_user_predictions", Path(args.out_predictions), index=False)

    pred_majority_share = float(max((pred_all == 0).mean(), (pred_all == 1).mean()))
    pred_prob_std = float(np.std(proba_all))
    collapse_risk_warning = ""
    if pred_majority_share >= 0.90 and pred_prob_std <= 0.08:
        collapse_risk_warning = (
            "Model collapse risk detected: predictions are dominated by one class "
            "with very low variation. Recheck target balance and training labels before trusting SHAP outputs."
        )
        print(f"[warning] {collapse_risk_warning}")

    report = {
        "target_source": target_meta.get("target_source", ""),
        "human_label_column": target_meta.get("human_label_column", ""),
        "human_label_rows": target_meta.get("human_label_rows", 0),
        "pseudo_label_rows": target_meta.get("pseudo_label_rows", 0),
        "joined_users": int(len(data)),
        "train_rows": int(len(X_train_model)),
        "test_rows": int(len(X_test)),
        "train_neg": int(neg),
        "train_pos": int(pos),
        "train_original_neg": int(balance_meta.get("original_neg", 0)),
        "train_original_pos": int(balance_meta.get("original_pos", 0)),
        "train_rebalance_applied": bool(balance_meta.get("applied", False)),
        "scale_pos_weight": float(scale_pos_weight),
        "cv_folds_requested": int(requested_cv_folds),
        "cv_folds_used": int(cv_folds_used),
        "early_stopping_rounds": int(args.early_stopping_rounds),
        "best_iteration": float(best_iteration_final) if not np.isnan(best_iteration_final) else np.nan,
        "cv_auc_mean": float(cv_auc_mean) if not np.isnan(cv_auc_mean) else np.nan,
        "cv_auc_std": float(cv_auc_std) if not np.isnan(cv_auc_std) else np.nan,
        "cv_best_iter_mean": float(cv_best_iter_mean) if not np.isnan(cv_best_iter_mean) else np.nan,
        "cv_best_iter_std": float(cv_best_iter_std) if not np.isnan(cv_best_iter_std) else np.nan,
        "accuracy": float(acc),
        "auc": float(auc) if not np.isnan(auc) else np.nan,
        "pred_majority_share": float(pred_majority_share),
        "pred_prob_std": float(pred_prob_std),
        "model_collapse_risk": bool(bool(collapse_risk_warning)),
        "model_artifact": str(model_out_path),
        "warning": " | ".join(
            [
                w
                for w in [
                    target_meta.get("warning", ""),
                    class_balance_warning,
                    collapse_risk_warning,
                ]
                if str(w).strip()
            ]
        ),
        "original_dims": prep_stats["original_dims"],
        "kept_dims": prep_stats["kept_dims"],
        "dropped_low_variance": prep_stats["dropped_low_variance"],
        "dropped_high_corr": prep_stats["dropped_high_corr"],
        "winsor_q_low": prep_stats["winsor_q_low"],
        "winsor_q_high": prep_stats["winsor_q_high"],
        "variance_threshold": prep_stats["variance_threshold"],
        "corr_drop_threshold": prep_stats["corr_drop_threshold"],
    }
    save_artifact_df(pd.DataFrame([report]), "xgb_target_report", Path(args.out_target_report), index=False)

    # Merged SHAP summary (duplicate semantic signals combined across embedding dims).
    def signal_name_from_feature(feat: str) -> str:
        lbl = str(emb_label_map.get(feat, "")).strip()
        if " - " in lbl:
            return lbl.split(" - ", 1)[1].strip()
        if lbl:
            return lbl
        return prettify_embedding_feature_name(feat)

    imp["signal_name"] = imp["feature"].apply(signal_name_from_feature)
    merged_imp = (
        imp.groupby("signal_name", as_index=False)
        .agg(
            mean_abs_shap=("mean_abs_shap", "sum"),
            dim_count=("feature", "nunique"),
            dimensions=("feature", lambda s: ", ".join(sorted(set(s.astype(str).tolist())))),
        )
        .sort_values("mean_abs_shap", ascending=False)
    )
    merged_path = Path(args.out_importance).with_name("xgb_embedding_feature_importance_merged.csv")
    save_artifact_df(merged_imp, "xgb_embedding_feature_importance_merged", merged_path, index=False)

    plot_df = merged_imp.sort_values("mean_abs_shap", ascending=True).copy()
    plot_df["label"] = plot_df["signal_name"] + " [" + plot_df["dim_count"].astype(str) + " dims]"
    fig_h = max(7.0, min(0.48 * len(plot_df) + 3.5, 40.0))
    plt.figure(figsize=(12, fig_h))
    plt.barh(plot_df["label"], plot_df["mean_abs_shap"], color="#8D6A3B")
    plt.xlabel("Total Mean |SHAP|")
    plt.ylabel("Merged Signal")
    plt.title("Merged SHAP Summary (Duplicate Embedding Signals Combined)")
    plt.tight_layout()
    plt.savefig(args.out_plot, dpi=180)
    plt.close()

    print(f"[saved] SHAP summary plot: {args.out_plot}")
    print(f"[saved] Feature importance CSV: {args.out_importance}")
    print(f"[saved] Merged feature importance CSV: {merged_path}")
    print(f"[saved] Target report CSV: {args.out_target_report}")
    print(f"[saved] User predictions CSV: {args.out_predictions}")
    print(
        "[preprocess] "
        f"original_dims={prep_stats['original_dims']} kept_dims={prep_stats['kept_dims']} "
        f"dropped_low_variance={prep_stats['dropped_low_variance']} dropped_high_corr={prep_stats['dropped_high_corr']}"
    )


if __name__ == "__main__":
    main()
