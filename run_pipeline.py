"""
Single-entry pipeline orchestrator with explicit stage ordering.

This script turns the project from ad-hoc scripts into a reproducible pipeline runner:
- deterministic ordered steps
- subprocess exit-code checks (fail fast)
- optional Redis publish step

Usage:
  python run_pipeline.py
  python run_pipeline.py --dry-run
  python run_pipeline.py --start-from train_user_behavior_gnn
  python run_pipeline.py --stop-after build_user_personas
    python run_pipeline.py --include-redis-publish
    python run_pipeline.py --no-redis-publish
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Step:
    id: str
    description: str
    cmd: list[str]
    optional: bool = False


def build_steps(include_redis_publish: bool) -> list[Step]:
    py = sys.executable
    steps = [
        Step(
            id="feature_engineering",
            description="Build user-level engineered feature matrix from raw CSVs",
            cmd=[py, "-m", "pipelines.preprocessing.feature_engineering"],
        ),
        Step(
            id="train_graphsage_user_embeddings",
            description="Train GraphSAGE embeddings from users/sessions",
            cmd=[py, "-m", "pipelines.training.train_graphsage_user_embeddings"],
        ),
        Step(
            id="build_gnn_nodes_from_flink",
            description="Build GNN node tables from Flink outputs",
            cmd=[py, "-m", "pipelines.preprocessing.build_gnn_nodes_from_flink"],
        ),
        Step(
            id="train_user_behavior_gnn",
            description="Train user behavior GNN and export GNN outputs + embeddings",
            cmd=[py, "-m", "pipelines.training.train_user_behavior_gnn"],
        ),
        Step(
            id="train_xgb_shap_sentiment",
            description="Train XGBoost + SHAP explainability on embeddings",
            cmd=[py, "-m", "pipelines.training.train_xgb_shap_sentiment", "--allow_pseudo_fallback"],
        ),
        Step(
            id="build_user_personas",
            description="Build user personas and SHAP explainability outputs",
            cmd=[py, "-m", "pipelines.training.build_user_personas"],
        ),
        Step(
            id="train_whatsapp_gru_mood_swings",
            description="Train GRU mood swing model",
            cmd=[py, "-m", "pipelines.training.train_whatsapp_gru_mood_swings"],
        ),
    ]

    if include_redis_publish:
        redis_url = os.getenv("REDIS_URL", "").strip()
        if not redis_url:
            raise ValueError("REDIS_URL is required when --include-redis-publish is set.")
        steps.append(
            Step(
                id="publish_dashboard_data_to_redis",
                description="Publish dashboard artifacts to Redis",
                cmd=[
                    py,
                    "-m",
                    "apps.tools.publish_dashboard_data_to_redis",
                    "--redis_url",
                    redis_url,
                    "--prefix",
                    os.getenv("MAYA_REDIS_PREFIX", "maya:dashboard"),
                ],
                optional=True,
            )
        )

    return steps


def slice_steps(steps: list[Step], start_from: str | None, stop_after: str | None) -> list[Step]:
    ids = [s.id for s in steps]
    start_idx = ids.index(start_from) if start_from else 0
    stop_idx = ids.index(stop_after) if stop_after else len(steps) - 1
    if start_idx > stop_idx:
        raise ValueError("--start-from must come before --stop-after in pipeline order.")
    return steps[start_idx : stop_idx + 1]


def run(steps: list[Step], dry_run: bool = False) -> int:
    total = len(steps)
    for i, step in enumerate(steps, start=1):
        cmd_str = " ".join(step.cmd)
        print(f"\n[{i}/{total}] {step.id}")
        print(f"  {step.description}")
        print(f"  $ {cmd_str}")
        if dry_run:
            continue

        res = subprocess.run(step.cmd, cwd=Path(__file__).resolve().parent)
        if res.returncode != 0:
            print(f"\n[FAIL] Step '{step.id}' exited with code {res.returncode}")
            return int(res.returncode)
        print(f"[OK] {step.id}")
    return 0


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run ordered Maya ML pipeline stages.")
    p.add_argument("--dry-run", action="store_true", help="Print steps without executing.")
    p.add_argument(
        "--include-redis-publish",
        action="store_true",
        help="Append Redis publish as final step (also auto-enabled when REDIS_URL is set).",
    )
    p.add_argument(
        "--no-redis-publish",
        action="store_true",
        help="Disable Redis publish step even when REDIS_URL is set.",
    )
    p.add_argument("--start-from", type=str, default=None, help="Start from this step id.")
    p.add_argument("--stop-after", type=str, default=None, help="Stop after this step id.")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    redis_url = os.getenv("REDIS_URL", "").strip()
    auto_include = bool(redis_url) and not bool(args.no_redis_publish)
    include_redis_publish = bool(args.include_redis_publish) or auto_include
    steps = build_steps(include_redis_publish=include_redis_publish)
    steps = slice_steps(steps, start_from=args.start_from, stop_after=args.stop_after)
    exit_code = run(steps, dry_run=bool(args.dry_run))
    raise SystemExit(exit_code)


if __name__ == "__main__":
    main()
