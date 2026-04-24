# Environment paths
# Prefer Windows venv layout when present, otherwise fall back to POSIX.
FLINK_PY  := $(if $(wildcard flink_venv/Scripts/python.exe),./flink_venv/Scripts/python.exe,./flink_venv/bin/python)
MAIN_PY   := $(if $(wildcard main_venv/Scripts/python.exe),./main_venv/Scripts/python.exe,./main_venv/bin/python)
FLINK_PIP := $(if $(wildcard flink_venv/Scripts/pip.exe),./flink_venv/Scripts/pip.exe,./flink_venv/bin/pip)
MAIN_PIP  := $(if $(wildcard main_venv/Scripts/pip.exe),./main_venv/Scripts/pip.exe,./main_venv/bin/pip)

# Tool paths
PYENV ?= pyenv

.PHONY: setup setup-all setup-flink-venv setup-main-venv start-dashboard start-flink start-producer start-all redis-publish redis-check empty-secret-data pipeline pipeline-cached pipeline-recompute gru-retrain docker-pipeline ingestor-sync pipeline-from-db

setup: setup-all

setup-all: setup-flink-venv setup-main-venv
	@echo "✓ Both environments ready"

setup-flink-venv:
	$(PYENV) install -s 3.10.14
	PYENV_VERSION=3.10.14 $(PYENV) exec python -m venv flink_venv
	$(FLINK_PIP) install --upgrade "pip<24.1" "setuptools<70" wheel
	$(FLINK_PIP) install apache-flink>=1.19.0 confluent-kafka torch transformers -c build-constraints.txt

setup-main-venv:
	$(PYENV) install -s 3.11.9
	PYENV_VERSION=3.11.9 $(PYENV) exec python -m venv main_venv
	$(MAIN_PIP) install --upgrade pip
	$(MAIN_PIP) install -r requirements.txt -c build-constraints.txt

start-dashboard:
	PYTHONPATH=. $(MAIN_PY) -m streamlit run apps/dashboard/streamlit_dashboard.py

start-flink:
	PYTHONPATH=. $(FLINK_PY) pipelines/streaming/flink_sentiment_job.py

start-producer:
	PYTHONPATH=. $(FLINK_PY) pipelines/ingestion/kafka_csv_producer.py --delay 0.1

start-all:
	@$(MAKE) start-producer &
	@$(MAKE) start-flink &
	@$(MAKE) start-dashboard

pipeline:
	PYTHONPATH=. $(MAIN_PY) run_pipeline.py $(FLAGS)

pipeline-cached:
	PYTHONPATH=. $(MAIN_PY) run_pipeline.py --use-cache $(FLAGS)

pipeline-recompute:
	PYTHONPATH=. $(MAIN_PY) run_pipeline.py --force-recompute $(FLAGS)

gru-retrain:
	PYTHONPATH=. $(MAIN_PY) -m pipelines.training.train_whatsapp_gru_mood_swings --force_retrain

ingestor-sync:
	@echo "[ingestor-sync] Triggering sync via $(INGESTOR_URL)/sync"
	@INGESTOR_URL="$(INGESTOR_URL)" $(MAIN_PY) -c "import json,os,sys,time,urllib.request; base=os.environ['INGESTOR_URL'].rstrip('/'); req=lambda p,m='GET': json.loads(urllib.request.urlopen(urllib.request.Request(base+p, method=m), timeout=10).read().decode() or '{}'); req('/sync','POST'); print('[ingestor-sync] Sync started; waiting for completion...'); code='deadline=time.time()+300\\nwhile True:\\n    status=req(\\'/status\\')\\n    if not status.get(\\'is_running\\', False):\\n        result=str(status.get(\\'last_result\\', \\'Unknown\\'))\\n        print(f\"[ingestor-sync] Completed with status: {result}\")\\n        sys.exit(0 if result.startswith(\\'Success\\') or result.startswith(\\'Partial Success\\') else 2)\\n    if time.time() > deadline:\\n        print(\\'[ingestor-sync] Timed out after 5 minutes waiting for /status\\')\\n        sys.exit(3)\\n    time.sleep(2)'; exec(code)"

pipeline-from-db: ingestor-sync
	PYTHONPATH=. $(MAIN_PY) run_pipeline.py $(FLAGS)

docker-pipeline:
	docker compose exec maya-app /app/main_venv/bin/python run_pipeline.py $(FLAGS)

# Legacy support / utility targets
PREFIX ?= maya:dashboard
INGESTOR_URL ?= http://localhost:8000
SECRET_DIR ?= secret_data

redis-publish:
	$(MAIN_PY) -m apps.tools.publish_dashboard_data_to_redis --prefix $(PREFIX)

redis-check:
	$(MAIN_PY) -m apps.tools.check_redis_publish --prefix $(PREFIX)

empty-secret-data:
	@mkdir -p $(SECRET_DIR)
	@find $(SECRET_DIR) -mindepth 1 -exec rm -rf {} +
