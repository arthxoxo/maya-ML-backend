from __future__ import annotations

import os
from pathlib import Path


def _env_path(name: str, default: Path) -> Path:
    raw = os.getenv(name, "").strip()
    return Path(raw) if raw else default


# Canonical project root for all stage scripts.
BASE_DIR = Path(__file__).resolve().parent

# Data/source directories.
SECRET_DATA_DIR = _env_path("MAYA_SECRET_DATA_DIR", BASE_DIR / "secret_data")
RAW_DATA_DIR = _env_path("MAYA_RAW_DATA_DIR", SECRET_DATA_DIR)
FLINK_ENGINEERED_DIR = _env_path("MAYA_FLINK_ENGINEERED_DIR", BASE_DIR / "flink_engineered")

# Pipeline stage IO directories.
GNN_PREPROCESSED_DIR = _env_path("MAYA_GNN_INPUT_DIR", BASE_DIR / "gnn_preprocessed")
GNN_OUTPUT_DIR = _env_path("MAYA_GNN_MODEL_OUTPUT_DIR", BASE_DIR / "gnn_outputs")
FEATURE_OUTPUT_DIR = _env_path("MAYA_FEATURE_OUTPUT_DIR", BASE_DIR)

# Organized artifact directories (CSV outputs).
ARTIFACTS_DIR = _env_path("MAYA_ARTIFACTS_DIR", BASE_DIR / "artifacts")
EMBEDDINGS_ARTIFACT_DIR = _env_path("MAYA_EMBEDDINGS_ARTIFACT_DIR", ARTIFACTS_DIR / "embeddings")
XGB_ARTIFACT_DIR = _env_path("MAYA_XGB_ARTIFACT_DIR", ARTIFACTS_DIR / "xgb")
PERSONA_ARTIFACT_DIR = _env_path("MAYA_PERSONA_ARTIFACT_DIR", ARTIFACTS_DIR / "persona")
SENTIMENT_ARTIFACT_DIR = _env_path("MAYA_SENTIMENT_ARTIFACT_DIR", ARTIFACTS_DIR / "sentiment")

# Online store settings.
REDIS_URL = os.getenv("REDIS_URL", "").strip()
REDIS_PREFIX = os.getenv("MAYA_REDIS_PREFIX", "maya:dashboard").strip() or "maya:dashboard"
STORE_TARGET = os.getenv("MAYA_STORE_TARGET", "file").strip().lower() or "file"
