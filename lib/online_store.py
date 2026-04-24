import json
import logging
import os
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import pandas as pd

from app_config import ARTIFACTS_DIR, REDIS_PREFIX, REDIS_URL, STORE_TARGET

logger = logging.getLogger(__name__)

MAX_PAYLOAD_SIZE = 2 * 1024 * 1024  # 2 MB limit for Redis payloads


def _is_truthy(value: str) -> bool:
    return str(value).strip().lower() in {"1", "true", "yes", "y", "on"}


def _backup_existing_csv(csv_path: Path, artifact_key: str) -> None:
    """
    Keep timestamped history copies before overwriting artifact CSVs.

    Controlled by env vars:
    - MAYA_ARTIFACT_BACKUP (default: 1)
    - MAYA_ARTIFACT_HISTORY_DIR (default: artifacts/history)
    """
    if not csv_path.exists():
        return

    backup_enabled = _is_truthy(os.getenv("MAYA_ARTIFACT_BACKUP", "1"))
    if not backup_enabled:
        return

    history_root = Path(os.getenv("MAYA_ARTIFACT_HISTORY_DIR", str(ARTIFACTS_DIR / "history")))
    ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    safe_key = artifact_key.replace(":", "_").replace("/", "_")
    backup_dir = history_root / safe_key
    backup_dir.mkdir(parents=True, exist_ok=True)

    backup_name = f"{csv_path.stem}_{ts}{csv_path.suffix}"
    backup_path = backup_dir / backup_name
    shutil.copy2(csv_path, backup_path)


def _get_redis_client():
    if STORE_TARGET not in {"redis", "hybrid", "auto"}:
        return None
    if not REDIS_URL:
        return None
    try:
        import redis  # type: ignore
    except Exception:
        logger.debug("redis-py not installed; skipping Redis connection.")
        return None

    try:
        client = redis.Redis.from_url(REDIS_URL, decode_responses=True)
        client.ping()
        return client
    except Exception as exc:
        logger.warning(f"Failed to connect to Redis at {REDIS_URL}: {exc}")
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
        except Exception as exc:
            logger.warning(f"Failed to load artifact '{artifact_key}' from Redis: {exc}")

    if fallback_csv_path.exists():
        return pd.read_csv(fallback_csv_path)

    if required:
        raise FileNotFoundError(
            f"Artifact '{artifact_key}' not found in Redis or CSV path: {fallback_csv_path}"
        )
    return pd.DataFrame()


def save_artifact_df(df: pd.DataFrame, artifact_key: str, csv_path: Path, index: bool = False) -> None:
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    _backup_existing_csv(csv_path, artifact_key)
    df.to_csv(csv_path, index=index)

    client = _get_redis_client()
    if client is None:
        return

    try:
        payload = json.dumps(df.to_dict(orient="records"), ensure_ascii=True)
        size_bytes = len(payload.encode("utf-8"))
        
        if size_bytes > MAX_PAYLOAD_SIZE:
            logger.warning(
                f"Redis payload for '{artifact_key}' too large ({size_bytes / 1024 / 1024:.2f} MB > {MAX_PAYLOAD_SIZE / 1024 / 1024:.2f} MB). "
                "Skipping Redis publish; file output remains authoritative."
            )
            return

        client.set(_redis_key(artifact_key), payload)
    except Exception as exc:
        # Keep file output authoritative; Redis publish is best effort.
        logger.warning(f"Failed to save artifact '{artifact_key}' to Redis: {exc}")


def save_artifact_file(artifact_key: str, file_path: Path) -> None:
    client = _get_redis_client()
    if client is None or not file_path.exists():
        return

    try:
        payload = json.dumps({"path": str(file_path), "exists": True}, ensure_ascii=True)
        client.set(_redis_key(artifact_key), payload)
    except Exception as exc:
        logger.warning(f"Failed to publish file reference '{artifact_key}' to Redis: {exc}")
