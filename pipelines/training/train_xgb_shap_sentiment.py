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
from online_store import load_artifact_df, save_artifact_df


def parse_args() -> argparse.Namespace:
    base = XGB_ARTIFACT_DIR
    p = argparse.ArgumentParser(description="Train XGBoost + SHAP on user embeddings for sentiment prediction")
    p.add_argument("--embeddings", type=str, default=str(EMBEDDINGS_ARTIFACT_DIR / "user_embeddings.csv"))
    p.add_argument("--sentiment", type=str, default=str(SENTIMENT_ARTIFACT_DIR / "sentiment_scores.csv"))
    p.add_argument("--sessions", type=str, default=str(SECRET_DATA_DIR / "sessions.csv"))
    p.add_argument("--feedback", type=str, default=str(SECRET_DATA_DIR / "feedbacks.csv"))
    p.add_argument(
        "--target_source",
        type=str,
        choices=["auto", "human", "pseudo"],
        default="auto",
        help="auto prefers human feedback labels, then falls back to pseudo sentiment labels",
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
    feedback_path = Path(args.feedback)

    meta = {
        "target_source": "",
        "human_label_column": "",
        "human_label_rows": 0,
        "pseudo_label_rows": 0,
        "warning": "",
    }

    human_targets = pd.DataFrame(columns=["user_id", "target", "target_label"])
    human_col = None
    human_rows = 0
    if feedback_path.exists():
        feedback = pd.read_csv(feedback_path)
        human_targets, human_col, human_rows = prepare_human_feedback_labels(feedback)

    if source_mode in {"auto", "human"} and not human_targets.empty:
        meta["target_source"] = "human_feedback"
        meta["human_label_column"] = str(human_col or "")
        meta["human_label_rows"] = int(human_rows)
        return human_targets[["user_id", "target"]].copy(), meta

    if source_mode == "human":
        raise ValueError(
            "target_source=human but no usable human feedback labels were found. "
            "Provide explicit feedback columns like rating/thumb/sentiment_label."
        )

    if source_mode == "auto" and not args.allow_pseudo_fallback:
        raise ValueError(
            "No usable human feedback labels found. Refusing pseudo-label fallback by default. "
            "Re-run with --allow_pseudo_fallback or provide explicit feedback labels."
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

    emb_path = Path(args.embeddings)
    sent_path = pick_sentiment_file(args.sentiment)
    sessions_path = Path(args.sessions)

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
    user_targets, target_meta = resolve_targets(args, sentiment, sessions_path=sessions_path)

    data = emb.merge(user_targets, on="user_id", how="inner")
    if data.empty:
        raise ValueError("Join result is empty. Check user_id alignment between embeddings and sentiment data")

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

    strat = y if y.nunique() > 1 else None
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=args.test_size, random_state=args.seed, stratify=strat
    )

    neg = int((y_train == 0).sum())
    pos = int((y_train == 1).sum())
    scale_pos_weight = (neg / pos) if pos > 0 else 1.0
    min_class_count = int(min(neg, pos))
    requested_cv_folds = max(2, int(args.cv_folds))
    cv_folds_used = int(min(requested_cv_folds, min_class_count)) if min_class_count >= 2 else 0

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
    )

    cv_auc_mean = float("nan")
    cv_auc_std = float("nan")
    cv_best_iter_mean = float("nan")
    cv_best_iter_std = float("nan")
    if cv_folds_used >= 2 and y_train.nunique() > 1:
        if cv_folds_used < requested_cv_folds:
            print(
                f"[warning] Requested {requested_cv_folds}-fold CV reduced to {cv_folds_used} due to minority class size."
            )
        cv = StratifiedKFold(n_splits=cv_folds_used, shuffle=True, random_state=args.seed)
        fold_aucs: list[float] = []
        fold_best_iters: list[float] = []
        for fold_idx, (tr_idx, va_idx) in enumerate(cv.split(X_train, y_train), start=1):
            X_tr_fold = X_train.iloc[tr_idx]
            y_tr_fold = y_train.iloc[tr_idx]
            X_va_fold = X_train.iloc[va_idx]
            y_va_fold = y_train.iloc[va_idx]

            model_fold = XGBClassifier(**model_params)
            model_fold.fit(
                X_tr_fold,
                y_tr_fold,
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
        train_strat = y_train if y_train.nunique() > 1 else None
        X_fit, X_eval, y_fit, y_eval = train_test_split(
            X_train,
            y_train,
            test_size=float(args.eval_size),
            random_state=args.seed,
            stratify=train_strat,
        )
        if y_eval.nunique() > 1 and len(X_eval) > 0:
            model.fit(
                X_fit,
                y_fit,
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
            model.fit(X_train, y_train, verbose=False)
            print("[fit] early stopping skipped: validation split had a single class.")
    except Exception as exc:
        model.fit(X_train, y_train, verbose=False)
        print(f"[warning] early stopping disabled due to split/fit issue: {exc}")

    pred = model.predict(X_test)
    proba = model.predict_proba(X_test)[:, 1]
    acc = accuracy_score(y_test, pred)
    auc = roc_auc_score(y_test, proba) if y_test.nunique() > 1 else float("nan")
    print(f"[metrics] accuracy={acc:.4f} auc={auc if np.isnan(auc) else round(auc, 4)}")
    print(f"[data] joined_users={len(data)} train={len(X_train)} test={len(X_test)}")
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

    report = {
        "target_source": target_meta.get("target_source", ""),
        "human_label_column": target_meta.get("human_label_column", ""),
        "human_label_rows": target_meta.get("human_label_rows", 0),
        "pseudo_label_rows": target_meta.get("pseudo_label_rows", 0),
        "joined_users": int(len(data)),
        "train_rows": int(len(X_train)),
        "test_rows": int(len(X_test)),
        "train_neg": int(neg),
        "train_pos": int(pos),
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
        "warning": " | ".join(
            [w for w in [target_meta.get("warning", ""), class_balance_warning] if str(w).strip()]
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
