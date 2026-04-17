from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

import pandas as pd

from app_config import REDIS_PREFIX, REDIS_URL, STORE_TARGET


def _get_redis_client():
    if STORE_TARGET not in {"redis", "hybrid", "auto"}:
        return None
    if not REDIS_URL:
        return None
    try:
        import redis  # type: ignore
    except Exception:
        return None

    try:
        client = redis.Redis.from_url(REDIS_URL, decode_responses=True)
        client.ping()
        return client
    except Exception:
        return None


def _redis_key(artifact_key: str) -> str:
    return f"{REDIS_PREFIX}:{artifact_key}"


def load_artifact_df(artifact_key: str, fallback_csv_path: Path, required: bool = True) -> pd.DataFrame:
    client = _get_redis_client()
    if client is not None:
        try:
            payload = client.get(_redis_key(artifact_key))
            if payload:
                rows = json.loads(payload)
                return pd.DataFrame(rows)
        except Exception:
            pass

    if fallback_csv_path.exists():
        return pd.read_csv(fallback_csv_path)

    if required:
        raise FileNotFoundError(
            f"Artifact '{artifact_key}' not found in Redis or CSV path: {fallback_csv_path}"
        )
    return pd.DataFrame()


def save_artifact_df(df: pd.DataFrame, artifact_key: str, csv_path: Path, index: bool = False) -> None:
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(csv_path, index=index)

    client = _get_redis_client()
    if client is None:
        return

    try:
        payload = json.dumps(df.to_dict(orient="records"), ensure_ascii=True)
        client.set(_redis_key(artifact_key), payload)
    except Exception:
        # Keep file output authoritative; Redis publish is best effort.
        pass


def save_artifact_file(artifact_key: str, file_path: Path) -> None:
    client = _get_redis_client()
    if client is None or not file_path.exists():
        return

    try:
        payload = json.dumps({"path": str(file_path), "exists": True}, ensure_ascii=True)
        client.set(_redis_key(artifact_key), payload)
    except Exception:
        pass
