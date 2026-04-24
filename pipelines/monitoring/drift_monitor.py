from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from app_config import BASE_DIR, EMBEDDINGS_ARTIFACT_DIR, MONITORING_ARTIFACT_DIR, SENTIMENT_ARTIFACT_DIR, XGB_ARTIFACT_DIR


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Stage 7 drift monitoring across pipeline runs")
    p.add_argument("--out_dir", type=str, default=str(MONITORING_ARTIFACT_DIR))
    p.add_argument("--latest_profile", type=str, default="drift_profile_latest.json")
    p.add_argument("--report_csv", type=str, default="drift_report.csv")
    p.add_argument("--summary_json", type=str, default="drift_summary.json")
    p.add_argument("--warn_threshold", type=float, default=2.0)
    p.add_argument("--alert_threshold", type=float, default=4.0)
    return p.parse_args()


def _source_files() -> dict[str, Path]:
    return {
        "feature_matrix": BASE_DIR / "user_feature_matrix.csv",
        "sentiment_scores": SENTIMENT_ARTIFACT_DIR / "sentiment_scores.csv",
        "xgb_predictions": XGB_ARTIFACT_DIR / "xgb_user_predictions.csv",
        "user_embeddings": EMBEDDINGS_ARTIFACT_DIR / "user_embeddings.csv",
        "gnn_scores": BASE_DIR / "gnn_outputs" / "user_behaviour_scores.csv",
    }


def _select_numeric_columns(df: pd.DataFrame) -> list[str]:
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    excluded = {"id", "user_id", "session_id"}
    return [c for c in numeric_cols if c.lower() not in excluded]


def _col_stats(series: pd.Series) -> dict[str, float | int]:
    vals = pd.to_numeric(series, errors="coerce").dropna().astype(float)
    if vals.empty:
        return {
            "n": 0,
            "mean": float("nan"),
            "std": float("nan"),
            "p05": float("nan"),
            "p50": float("nan"),
            "p95": float("nan"),
        }

    return {
        "n": int(vals.shape[0]),
        "mean": float(vals.mean()),
        "std": float(vals.std(ddof=0)),
        "p05": float(vals.quantile(0.05)),
        "p50": float(vals.quantile(0.50)),
        "p95": float(vals.quantile(0.95)),
    }


def build_current_profile() -> dict:
    profile = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "sources": {},
    }

    for source_name, path in _source_files().items():
        if not path.exists():
            profile["sources"][source_name] = {
                "path": str(path),
                "exists": False,
                "rows": 0,
                "columns": {},
            }
            continue

        df = pd.read_csv(path)
        numeric_cols = _select_numeric_columns(df)
        col_stats = {col: _col_stats(df[col]) for col in numeric_cols}
        profile["sources"][source_name] = {
            "path": str(path),
            "exists": True,
            "rows": int(len(df)),
            "columns": col_stats,
        }

    return profile


def _safe_float(value: object) -> float:
    try:
        x = float(value)
        return x
    except Exception:
        return float("nan")


def drift_score(prev: dict, cur: dict) -> float:
    eps = 1e-6

    prev_mean = _safe_float(prev.get("mean"))
    cur_mean = _safe_float(cur.get("mean"))
    prev_std = abs(_safe_float(prev.get("std")))
    cur_std = abs(_safe_float(cur.get("std")))
    prev_p95 = _safe_float(prev.get("p95"))
    cur_p95 = _safe_float(cur.get("p95"))

    mean_z = abs(cur_mean - prev_mean) / max(prev_std, eps)
    std_log_ratio = abs(np.log((cur_std + eps) / (prev_std + eps)))
    p95_shift = abs(cur_p95 - prev_p95) / max(abs(prev_p95), 1.0)

    score = 0.6 * min(mean_z, 10.0) + 0.3 * min(std_log_ratio * 5.0, 10.0) + 0.1 * min(p95_shift, 10.0)
    return float(score)


def compare_profiles(prev_profile: dict, cur_profile: dict, warn_threshold: float, alert_threshold: float) -> tuple[pd.DataFrame, dict]:
    rows: list[dict] = []

    prev_sources = prev_profile.get("sources", {})
    cur_sources = cur_profile.get("sources", {})

    for source_name, cur_src in cur_sources.items():
        prev_src = prev_sources.get(source_name, {})
        cur_cols = cur_src.get("columns", {})
        prev_cols = prev_src.get("columns", {})

        for col_name, cur_stats in cur_cols.items():
            prev_stats = prev_cols.get(col_name)
            if not prev_stats:
                rows.append(
                    {
                        "source": source_name,
                        "column": col_name,
                        "status": "new_column",
                        "drift_score": float("nan"),
                        "current_mean": _safe_float(cur_stats.get("mean")),
                        "previous_mean": float("nan"),
                        "current_std": _safe_float(cur_stats.get("std")),
                        "previous_std": float("nan"),
                        "current_rows": int(cur_src.get("rows", 0)),
                        "previous_rows": int(prev_src.get("rows", 0)),
                    }
                )
                continue

            score = drift_score(prev_stats, cur_stats)
            if score >= alert_threshold:
                status = "alert"
            elif score >= warn_threshold:
                status = "warn"
            else:
                status = "ok"

            rows.append(
                {
                    "source": source_name,
                    "column": col_name,
                    "status": status,
                    "drift_score": score,
                    "current_mean": _safe_float(cur_stats.get("mean")),
                    "previous_mean": _safe_float(prev_stats.get("mean")),
                    "current_std": _safe_float(cur_stats.get("std")),
                    "previous_std": _safe_float(prev_stats.get("std")),
                    "current_rows": int(cur_src.get("rows", 0)),
                    "previous_rows": int(prev_src.get("rows", 0)),
                }
            )

    report_df = pd.DataFrame(rows)
    if report_df.empty:
        summary = {
            "overall_status": "baseline",
            "compared_columns": 0,
            "alerts": 0,
            "warnings": 0,
            "ok": 0,
            "note": "No previous profile available or no overlapping columns to compare.",
        }
        return report_df, summary

    alerts = int((report_df["status"] == "alert").sum())
    warns = int((report_df["status"] == "warn").sum())
    oks = int((report_df["status"] == "ok").sum())

    overall = "alert" if alerts > 0 else ("warn" if warns > 0 else "ok")
    summary = {
        "overall_status": overall,
        "compared_columns": int((report_df["status"].isin(["ok", "warn", "alert"])).sum()),
        "alerts": alerts,
        "warnings": warns,
        "ok": oks,
    }
    return report_df, summary


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def main() -> None:
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    latest_profile_path = out_dir / args.latest_profile
    report_path = out_dir / args.report_csv
    summary_path = out_dir / args.summary_json

    current_profile = build_current_profile()

    prev_profile = {}
    if latest_profile_path.exists():
        try:
            prev_profile = json.loads(latest_profile_path.read_text(encoding="utf-8"))
        except Exception:
            prev_profile = {}

    report_df, summary = compare_profiles(
        prev_profile=prev_profile,
        cur_profile=current_profile,
        warn_threshold=float(args.warn_threshold),
        alert_threshold=float(args.alert_threshold),
    )

    generated_at = datetime.now(timezone.utc)
    summary_payload = {
        "generated_at_utc": generated_at.isoformat(),
        "summary": summary,
        "current_profile_path": str(latest_profile_path),
        "report_csv_path": str(report_path),
    }

    if report_df.empty:
        report_df = pd.DataFrame(
            [
                {
                    "source": "baseline",
                    "column": "n/a",
                    "status": "baseline",
                    "drift_score": float("nan"),
                    "current_mean": float("nan"),
                    "previous_mean": float("nan"),
                    "current_std": float("nan"),
                    "previous_std": float("nan"),
                    "current_rows": 0,
                    "previous_rows": 0,
                }
            ]
        )

    report_df.to_csv(report_path, index=False)
    write_json(summary_path, summary_payload)

    ts = generated_at.strftime("%Y%m%d_%H%M%S")
    historical_profile = out_dir / f"drift_profile_{ts}.json"
    write_json(historical_profile, current_profile)
    write_json(latest_profile_path, current_profile)

    print(f"[ok] Drift report saved: {report_path}")
    print(f"[ok] Drift summary saved: {summary_path}")
    print(f"[ok] Current profile saved: {latest_profile_path}")
    print(f"[ok] Historical profile snapshot: {historical_profile}")
    print(
        "[ok] Drift status: "
        f"{summary_payload['summary'].get('overall_status', 'unknown')} "
        f"(alerts={summary_payload['summary'].get('alerts', 0)}, "
        f"warnings={summary_payload['summary'].get('warnings', 0)})"
    )


if __name__ == "__main__":
    main()
