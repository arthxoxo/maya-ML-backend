"""
Create user personas from GraphSAGE embeddings and explain persona assignment.

Pipeline:
1) Cluster user embeddings with KMeans (default k=5)
2) Join clusters with original user data and sentiment aggregates
3) Train RandomForest to predict persona_id from original user columns
4) Produce per-user top-3 reasons (feature-level explanation proxy)

Outputs:
  - user_persona_table.csv: user_id, persona_label, top 3 reasons
  - persona_profiles.csv: persona-level summary stats and names
  - persona_feature_importance.csv: global RF feature importance

Usage:
  /path/to/python build_user_personas.py
  /path/to/python build_user_personas.py --embeddings user_embeddings.csv --users secret_data/users.csv
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestClassifier

from app_config import EMBEDDINGS_ARTIFACT_DIR, PERSONA_ARTIFACT_DIR, SECRET_DATA_DIR, SENTIMENT_ARTIFACT_DIR
from online_store import load_artifact_df, save_artifact_df, save_artifact_file


def parse_args() -> argparse.Namespace:
    base = PERSONA_ARTIFACT_DIR
    p = argparse.ArgumentParser(description="Build user personas from embeddings and explain with RandomForest")
    p.add_argument("--embeddings", type=str, default=str(EMBEDDINGS_ARTIFACT_DIR / "user_embeddings.csv"))
    p.add_argument("--users", type=str, default=str(SECRET_DATA_DIR / "users.csv"))
    p.add_argument("--sentiment", type=str, default=str(SENTIMENT_ARTIFACT_DIR / "sentiment_scores.csv"))
    p.add_argument("--sessions", type=str, default=str(SECRET_DATA_DIR / "sessions.csv"))
    p.add_argument("--k", type=int, default=5)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--out_table", type=str, default=str(base / "user_persona_table.csv"))
    p.add_argument("--out_profiles", type=str, default=str(base / "persona_profiles.csv"))
    p.add_argument("--out_importance", type=str, default=str(base / "persona_feature_importance.csv"))
    p.add_argument("--out_shap_plot", type=str, default=str(base / "persona_shap_summary.png"))
    p.add_argument("--out_user_shap", type=str, default=str(base / "persona_user_feature_contributions.csv"))
    return p.parse_args()


def norm_label(lbl: str) -> str:
    s = str(lbl).strip().lower()
    if s in {"positive", "pos", "1", "true"}:
        return "positive"
    if s in {"negative", "neg", "0", "false"}:
        return "negative"
    return "neutral"


def load_sentiment_user_level(sentiment_path: Path, sessions_path: Path) -> pd.DataFrame:
    if not sentiment_path.exists():
        return pd.DataFrame(columns=["user_id", "avg_sentiment", "dominant_sentiment", "msg_count"])

    s = pd.read_csv(sentiment_path)
    if "role" in s.columns:
        role = s["role"].astype(str).str.lower().str.strip()
        if role.isin(["user"]).any():
            # Use only user-originated messages for user sentiment profiling.
            s = s[role.eq("user")].copy()
    if "sentiment_label" not in s.columns and "sentiment" in s.columns:
        s["sentiment_label"] = s["sentiment"]
    if "sentiment_label" not in s.columns:
        s["sentiment_label"] = "neutral"

    if "user_id" not in s.columns:
        if "session_id" in s.columns and sessions_path.exists():
            sess = pd.read_csv(sessions_path, usecols=["id", "user_id"])
            sess["id"] = pd.to_numeric(sess["id"], errors="coerce")
            sess["user_id"] = pd.to_numeric(sess["user_id"], errors="coerce")
            s["session_id"] = pd.to_numeric(s["session_id"], errors="coerce")
            s = s.merge(sess.rename(columns={"id": "session_id"}), on="session_id", how="left")
        else:
            return pd.DataFrame(columns=["user_id", "avg_sentiment", "dominant_sentiment", "msg_count"])

    s["user_id"] = pd.to_numeric(s["user_id"], errors="coerce")
    s = s.dropna(subset=["user_id"]).copy()
    s["user_id"] = s["user_id"].astype(int)

    if "sentiment_score" in s.columns:
        s["sentiment_score"] = pd.to_numeric(s["sentiment_score"], errors="coerce").fillna(0.0)
    else:
        s["sentiment_score"] = 0.0

    s["sentiment_label_norm"] = s["sentiment_label"].map(norm_label)
    agg = s.groupby("user_id", as_index=False).agg(
        avg_sentiment=("sentiment_score", "mean"),
        msg_count=("sentiment_score", "size"),
        pos_ratio=("sentiment_label_norm", lambda x: float((x == "positive").mean())),
        neg_ratio=("sentiment_label_norm", lambda x: float((x == "negative").mean())),
    )

    dom = (
        s.groupby(["user_id", "sentiment_label_norm"]).size().reset_index(name="cnt")
        .sort_values(["user_id", "cnt"], ascending=[True, False])
        .drop_duplicates("user_id")
        .rename(columns={"sentiment_label_norm": "dominant_sentiment"})[["user_id", "dominant_sentiment"]]
    )

    return agg.merge(dom, on="user_id", how="left")


def preprocess_users(users: pd.DataFrame) -> pd.DataFrame:
    u = users.copy()

    id_col = "id" if "id" in u.columns else "user_id"
    u["user_id"] = pd.to_numeric(u[id_col], errors="coerce")
    u = u.dropna(subset=["user_id"]).copy()
    u["user_id"] = u["user_id"].astype(int)

    u["created_at"] = pd.to_datetime(u.get("created_at"), errors="coerce", utc=True)
    u["account_age_days"] = (pd.Timestamp.now(tz="UTC") - u["created_at"]).dt.total_seconds() / 86400.0
    u["account_age_days"] = pd.to_numeric(u["account_age_days"], errors="coerce").fillna(0.0).clip(lower=0)

    if "contacts_backfilled" in u.columns:
        u["contacts_backfilled"] = (
            u["contacts_backfilled"].astype(str).str.lower().isin(["true", "t", "1", "yes", "y"])
        ).astype(int)
    else:
        u["contacts_backfilled"] = 0

    for c in ["status", "type"]:
        if c not in u.columns:
            u[c] = "unknown"
        u[c] = u[c].fillna("unknown").astype(str)

    return u[[
        "user_id",
        "account_age_days",
        "contacts_backfilled",
        "status",
        "type",
    ]]


def persona_name(row: pd.Series, age_q2: float, msg_q2: float) -> str:
    s = float(row.get("avg_sentiment", 0.0))
    neg_ratio = float(row.get("neg_ratio", 0.0))
    pos_ratio = float(row.get("pos_ratio", 0.0))
    age = float(row.get("account_age_days", 0.0))
    msgs = float(row.get("msg_count", 0.0))

    if s < -0.03 or neg_ratio >= 0.33:
        mood = "Frustrated"
    elif s > 0.12 and pos_ratio >= neg_ratio:
        mood = "Satisfied"
    else:
        mood = "Neutral"

    tenure = "Long-term" if age >= age_q2 else "New"
    engagement = "Highly Active" if msgs >= msg_q2 else "Low Activity"
    return f"{mood} {tenure} {engagement} Users"


def behavior_reason(feat: str, row: pd.Series, medians: dict[str, float]) -> str:
    if feat == "msg_count":
        return "high message volume" if float(row.get("msg_count", 0.0)) >= medians.get("msg_count", 0.0) else "low message volume"
    if feat == "avg_sentiment":
        val = float(row.get("avg_sentiment", 0.0))
        if val <= -0.12:
            return "negative sentiment"
        if val >= 0.12:
            return "positive sentiment"
        return "neutral sentiment"
    if feat == "pos_ratio":
        return "frequent positive messages" if float(row.get("pos_ratio", 0.0)) >= medians.get("pos_ratio", 0.0) else "limited positive messages"
    if feat == "neg_ratio":
        return "frequent negative messages" if float(row.get("neg_ratio", 0.0)) >= medians.get("neg_ratio", 0.0) else "limited negative messages"
    if feat == "account_age_days":
        return "long account tenure" if float(row.get("account_age_days", 0.0)) >= medians.get("account_age_days", 0.0) else "newer account tenure"
    if feat == "contacts_backfilled":
        return "contacts synced" if float(row.get("contacts_backfilled", 0.0)) >= 0.5 else "contacts not synced"
    if feat.startswith("dominant_sentiment_"):
        lbl = feat.split("dominant_sentiment_", 1)[1]
        return f"dominant {lbl} tone"
    if feat.startswith("status_"):
        return f"account status: {feat.split('status_', 1)[1]}"
    if feat.startswith("type_"):
        return f"user type: {feat.split('type_', 1)[1]}"
    return feat.replace("_", " ")


def main() -> None:
    args = parse_args()

    emb = load_artifact_df("user_embeddings", Path(args.embeddings))
    if "user_id" not in emb.columns:
        raise ValueError("Embeddings file must include user_id")
    emb["user_id"] = pd.to_numeric(emb["user_id"], errors="coerce")
    emb = emb.dropna(subset=["user_id"]).copy()
    emb["user_id"] = emb["user_id"].astype(int)

    emb_cols = [c for c in emb.columns if str(c).startswith("emb_")]
    if not emb_cols:
        raise ValueError("No embedding columns found (expected emb_*)")
    emb[emb_cols] = emb[emb_cols].apply(pd.to_numeric, errors="coerce").fillna(0.0)

    km = KMeans(n_clusters=args.k, random_state=args.seed, n_init=20)
    emb["persona_id"] = km.fit_predict(emb[emb_cols])

    users_raw = load_artifact_df("users", Path(args.users))
    users = preprocess_users(users_raw)
    sent = load_sentiment_user_level(Path(args.sentiment), Path(args.sessions))

    base = emb[["user_id", "persona_id"]].merge(users, on="user_id", how="left")
    base = base.merge(sent, on="user_id", how="left")
    base["avg_sentiment"] = pd.to_numeric(base.get("avg_sentiment"), errors="coerce").fillna(0.0)
    base["msg_count"] = pd.to_numeric(base.get("msg_count"), errors="coerce").fillna(0.0)
    base["dominant_sentiment"] = base.get("dominant_sentiment", "neutral").fillna("neutral")

    # Persona naming using cluster profile summaries.
    profiles = (
        base.groupby("persona_id", as_index=False)
        .agg(
            users=("user_id", "nunique"),
            avg_sentiment=("avg_sentiment", "mean"),
            account_age_days=("account_age_days", "mean"),
            msg_count=("msg_count", "mean"),
            pos_ratio=("pos_ratio", "mean"),
            neg_ratio=("neg_ratio", "mean"),
        )
    )
    age_q2 = float(base["account_age_days"].median()) if len(base) else 0.0
    msg_q2 = float(base["msg_count"].median()) if len(base) else 0.0
    profiles["persona_label"] = profiles.apply(lambda r: persona_name(r, age_q2, msg_q2), axis=1)

    base = base.merge(profiles[["persona_id", "persona_label"]], on="persona_id", how="left")

    # RandomForest + SHAP explainability from behavior-focused features only.
    model_df = base.copy()
    num_cols = ["account_age_days", "contacts_backfilled", "avg_sentiment", "msg_count", "pos_ratio", "neg_ratio"]
    for c in num_cols:
        if c not in model_df.columns:
            model_df[c] = 0.0
        model_df[c] = pd.to_numeric(model_df[c], errors="coerce").fillna(0.0)

    cat_cols = ["status", "type", "dominant_sentiment"]
    for c in cat_cols:
        if c not in model_df.columns:
            model_df[c] = "unknown"
        model_df[c] = model_df[c].fillna("unknown").astype(str)

    X_num = model_df[num_cols].copy()
    X_cat = pd.get_dummies(model_df[cat_cols], prefix=cat_cols, drop_first=False)
    X = pd.concat([X_num, X_cat], axis=1)
    y = model_df["persona_id"].astype(int)

    rf = RandomForestClassifier(n_estimators=500, random_state=args.seed, class_weight="balanced")
    rf.fit(X, y)

    explainer = shap.TreeExplainer(rf)
    shap_vals = explainer.shap_values(X)
    if isinstance(shap_vals, list):
        # list[class] -> (n_samples, n_features), aggregate classes to 2D
        shap_2d = np.mean(np.abs(np.stack(shap_vals, axis=0)), axis=0)
    else:
        shap_vals = np.array(shap_vals)
        if shap_vals.ndim == 3:
            # Common multiclass shape: (n_samples, n_features, n_classes)
            shap_2d = np.mean(np.abs(shap_vals), axis=2)
        else:
            shap_2d = np.abs(shap_vals)

    mean_abs_shap = shap_2d.mean(axis=0)
    fi = pd.DataFrame({"feature": X.columns, "importance": mean_abs_shap}).sort_values("importance", ascending=False)
    save_artifact_df(fi, "persona_feature_importance", Path(args.out_importance), index=False)

    plt.figure(figsize=(10, 6))
    shap.summary_plot(shap_2d, X, show=False, max_display=20)
    plt.tight_layout()
    plt.savefig(Path(args.out_shap_plot), dpi=180)
    plt.close()

    medians = {c: float(model_df[c].median()) if c in model_df.columns else 0.0 for c in num_cols}

    # Per-user local SHAP contributions for exact individual-level analysis.
    user_shap_rows = []
    abs_shap = np.abs(shap_2d)
    for i in range(len(model_df)):
        uid = int(model_df.iloc[i]["user_id"])
        plabel = str(model_df.iloc[i]["persona_label"])
        pid = int(model_df.iloc[i]["persona_id"])
        for j, f in enumerate(X.columns):
            sval = float(shap_2d[i, j])
            user_shap_rows.append(
                {
                    "user_id": uid,
                    "persona_id": pid,
                    "persona_label": plabel,
                    "feature": str(f),
                    "shap_value": sval,
                    "abs_shap": float(abs_shap[i, j]),
                }
            )
    user_shap_df = pd.DataFrame(user_shap_rows)
    save_artifact_df(user_shap_df, "persona_user_feature_contributions", Path(args.out_user_shap), index=False)

    reason_rows = []
    for i in range(len(model_df)):
        row = model_df.iloc[i]
        uid = int(row["user_id"])
        label = str(row["persona_label"])

        local = user_shap_df[user_shap_df["user_id"] == uid].sort_values("abs_shap", ascending=False)
        top_feats = local["feature"].head(3).tolist()
        top3 = [behavior_reason(f, row, medians) for f in top_feats]
        while len(top3) < 3:
            top3.append("n/a")

        reason_rows.append(
            {
                "user_id": uid,
                "persona_label": label,
                "top_reason_1": top3[0],
                "top_reason_2": top3[1],
                "top_reason_3": top3[2],
            }
        )

    user_table = pd.DataFrame(reason_rows).sort_values("user_id")
    save_artifact_df(user_table, "user_persona_table", Path(args.out_table), index=False)
    save_artifact_df(profiles, "persona_profiles", Path(args.out_profiles), index=False)
    save_artifact_file("persona_shap_summary", Path(args.out_shap_plot))

    print(f"Saved user persona table: {args.out_table}")
    print(f"Saved persona profiles: {args.out_profiles}")
    print(f"Saved persona feature importance: {args.out_importance}")
    print(f"Saved persona SHAP summary plot: {args.out_shap_plot}")
    print(f"Saved per-user local SHAP contributions: {args.out_user_shap}")


if __name__ == "__main__":
    main()
