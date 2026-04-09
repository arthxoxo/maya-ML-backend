from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import pandas as pd

from config import BASE_DIR, REDIS_PREFIX


DEFAULT_DATASETS = {
    "user_behaviour_scores": "gnn_outputs/user_behaviour_scores.csv",
    "user_feature_importance_global": "gnn_outputs/user_feature_importance_global.csv",
    "user_feature_importance_per_user": "gnn_outputs/user_feature_importance_per_user.csv",
    "user_embeddings": "artifacts/embeddings/user_embeddings.csv",
    "xgb_embedding_feature_importance": "artifacts/xgb/xgb_embedding_feature_importance.csv",
    "xgb_target_report": "artifacts/xgb/xgb_target_report.csv",
    "xgb_user_predictions": "artifacts/xgb/xgb_user_predictions.csv",
    "embedding_dimension_labels": "artifacts/embeddings/embedding_dimension_labels.csv",
    "user_persona_table": "artifacts/persona/user_persona_table.csv",
    "persona_profiles": "artifacts/persona/persona_profiles.csv",
    "persona_feature_importance": "artifacts/persona/persona_feature_importance.csv",
    "persona_user_feature_contributions": "artifacts/persona/persona_user_feature_contributions.csv",
    "users_nodes": "gnn_preprocessed/users_nodes.csv",
}


def parse_args() -> argparse.Namespace:
    base = BASE_DIR
    p = argparse.ArgumentParser(description="Publish dashboard CSV outputs to Redis as JSON records")
    p.add_argument("--base_dir", type=str, default=str(base), help="Repository root directory")
    p.add_argument("--redis_url", type=str, default=os.getenv("REDIS_URL", "redis://localhost:6379/0"))
    p.add_argument("--prefix", type=str, default=os.getenv("MAYA_REDIS_PREFIX", REDIS_PREFIX))
    p.add_argument(
        "--strict",
        action="store_true",
        help="Fail when any configured dataset is missing; default skips missing files",
    )
    return p.parse_args()


def publish_dataset(client, key: str, file_path: Path, prefix: str) -> tuple[bool, int]:
    if not file_path.exists():
        return False, 0

    df = pd.read_csv(file_path)
    records = df.to_dict(orient="records")
    payload = json.dumps(records, ensure_ascii=True)
    client.set(f"{prefix}:{key}", payload)
    return True, len(df)


def main() -> None:
    args = parse_args()

    try:
        import redis  # type: ignore
    except Exception as exc:
        raise RuntimeError("redis package not installed. Run: pip install redis") from exc

    base_dir = Path(args.base_dir)
    client = redis.Redis.from_url(args.redis_url, decode_responses=True)
    client.ping()

    published = 0
    skipped = 0
    total_rows = 0

    for key, rel_path in DEFAULT_DATASETS.items():
        path = base_dir / rel_path
        ok, n_rows = publish_dataset(client, key=key, file_path=path, prefix=args.prefix)
        if ok:
            published += 1
            total_rows += n_rows
            print(f"[published] key={args.prefix}:{key} rows={n_rows} file={path}")
        else:
            skipped += 1
            msg = f"[missing] key={args.prefix}:{key} file={path}"
            if args.strict:
                raise FileNotFoundError(msg)
            print(msg)

    print(f"[done] published={published} skipped={skipped} total_rows={total_rows}")


if __name__ == "__main__":
    main()
