# maya-ML-backend

MLops-style feature engineering and GNN user behavior analysis.

## Project Structure

```text
apps/
	dashboard/            # Streamlit UI
	tools/                # utility apps (Redis publisher)
pipelines/
	ingestion/            # CSV -> Kafka producers
	streaming/            # Flink jobs
	preprocessing/        # Flink outputs -> ML node/features
	training/             # GNN, GraphSAGE, XGB+SHAP, personas
config.py               # global settings
online_store.py         # Redis/file artifact IO
artifacts/              # organized CSV/plot outputs
  embeddings/
  xgb/
  persona/
  sentiment/
```

## Centralized Config + Online Store

The project now uses shared infrastructure modules:

- `app_config.py`: single source of truth for all paths and environment settings.
- `online_store.py`: file + Redis artifact layer used by training and persona scripts.

Key environment variables:

- `MAYA_RAW_DATA_DIR` (default: `secret_data/`)
- `MAYA_SECRET_DATA_DIR` (default: `secret_data/`)
- `MAYA_GNN_INPUT_DIR` (default: `gnn_preprocessed/`)
- `MAYA_GNN_MODEL_OUTPUT_DIR` (default: `gnn_outputs/`)
- `MAYA_FEATURE_OUTPUT_DIR` (default: repo root)
- `MAYA_STORE_TARGET` (`file`, `redis`, `hybrid`, or `auto`; default is `auto`)
- `REDIS_URL` (required for Redis mode)
- `MAYA_REDIS_PREFIX` (default: `maya:dashboard`)

Connected stage behavior:

- Preprocessing node-builder and training stages now publish generated CSV outputs directly to Redis when `REDIS_URL` is set.
- GNN stage reads node artifacts from Redis-first (with CSV fallback) and writes outputs to both CSV and Redis.
- XGBoost + SHAP stage reads embeddings from the shared artifact store and publishes SHAP outputs back to Redis and CSV.
- Persona stage reads embeddings from the shared artifact store and publishes persona outputs back to Redis and CSV.

This removes isolated CSV-only stages and creates an online, artifact-connected pipeline.

## Streaming Flow

This project now supports the requested pipeline:

1. CSV -> Kafka
2. Kafka -> Flink transforms
3. Flink -> filesystem engineered datasets
4. Flink datasets -> GNN node tables
5. GNN training on node tables

### 1) Produce CSV data into Kafka topics

```bash
python -m pipelines.ingestion.kafka_csv_producer --broker localhost:9092 --delay 0.1
```

Raw topics produced:

- `maya_users`
- `maya_sessions`
- `maya_feedbacks`
- `maya_whatsapp_messages`

### 2) Run Flink sentiment job (Kafka -> Kafka)

```bash
python -m pipelines.streaming.flink_sentiment_job
```

Consumes raw Kafka topics:

- `maya_users`
- `maya_sessions`
- `maya_feedbacks`
- `maya_whatsapp_messages`

Produces Flink filesystem datasets under `flink_engineered/`:

- `flink_engineered/users`
- `flink_engineered/sessions`
- `flink_engineered/feedbacks`
- `flink_engineered/messages_sentiment`

### 3) Build GNN node tables from Flink outputs

```bash
python -m pipelines.preprocessing.build_gnn_nodes_from_flink
```

Path note:
- Canonical Flink path is `flink_engineered/*`.
- For backward compatibility, the builder also accepts legacy `engineered_features/*` if present.

This writes:

- `gnn_preprocessed/users_nodes.csv`
- `gnn_preprocessed/sessions_nodes.csv`
- `gnn_preprocessed/messages_nodes.csv`
- `gnn_preprocessed/feedback_nodes.csv`

### 4) Train GNN

```bash
python -m pipelines.training.train_user_behavior_gnn
```

Outputs:

- `gnn_outputs/user_behaviour_scores.csv`
- `gnn_outputs/user_feature_importance_global.csv`
- `gnn_outputs/user_feature_importance_per_user.csv`
- `gnn_outputs/user_embeddings.csv`

### 5) XGBoost + SHAP explainability on embeddings (Stage 6)

```bash
python -m pipelines.training.train_xgb_shap_sentiment \
	--embeddings user_embeddings.csv \
	--sentiment sentiment_scores.csv \
	--feedback secret_data/feedbacks.csv
```

By default, this step prefers human-supervised labels from feedback data.
If no usable human labels are present, the script fails instead of silently training on pseudo labels.

To explicitly allow pseudo-label fallback (noisy target), run:

```bash
python -m pipelines.training.train_xgb_shap_sentiment --allow_pseudo_fallback
```

Artifacts:

- `artifacts/xgb/shap_summary.png`
- `artifacts/xgb/xgb_embedding_feature_importance.csv`
- `artifacts/xgb/xgb_target_report.csv` (target provenance + warning + model metrics)

### 6) Redis Backfill (Optional) and Dashboard

Install Redis client package once in your environment:

```bash
pip install redis
```

If Redis is not running locally, start it quickly with Docker:

```bash
docker run --name maya-redis -p 6379:6379 -d redis:7
```

Backfill Redis keys from existing CSVs (optional, useful when migrating historical files):

```bash
python -m apps.tools.publish_dashboard_data_to_redis \
	--redis_url redis://localhost:6379/0 \
	--prefix maya:dashboard
```

Run Streamlit with Redis enabled:

```bash
export REDIS_URL=redis://localhost:6379/0
export MAYA_REDIS_PREFIX=maya:dashboard
python -m streamlit run apps/dashboard/streamlit_dashboard.py
```

Dashboard behavior:

- Reads from Redis first for core datasets.
- Falls back to CSV files if Redis is unavailable or keys are missing.

Quick verification commands:

```bash
export REDIS_URL='redis://default:<password>@redis-17723.crce206.ap-south-1-1.ec2.cloud.redislabs.com:17723/0'
make redis-publish
make redis-check
```

## One-Command Ordered Pipeline

Use the orchestrator to run stages in code-defined order with fail-fast exit checks:

```bash
python run_pipeline.py
```

Useful options:

- `--dry-run` (print order only)
- `--start-from <step_id>`
- `--stop-after <step_id>`
- `--include-redis-publish` (force append Redis backfill step)
- `--no-redis-publish` (disable Redis backfill step)

By default, if `REDIS_URL` is set, `run_pipeline.py` auto-appends the Redis backfill step.
