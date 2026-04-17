# Environment paths
FLINK_PY  := ./flink_venv/bin/python
MAIN_PY   := ./main_venv/bin/python

# Tool paths
PYENV ?= pyenv

.PHONY: setup setup-all setup-flink-venv setup-main-venv start-dashboard start-flink start-producer start-all redis-publish redis-check setup-dev reset-data pipeline docker-pipeline

setup: setup-all

setup-all: setup-flink-venv setup-main-venv
	@echo "✓ Both environments ready"

setup-flink-venv:
	$(PYENV) install -s 3.10.14
	PYENV_VERSION=3.10.14 $(PYENV) exec python -m venv flink_venv
	./flink_venv/bin/pip install --upgrade "pip<24.1" "setuptools<70" wheel
	./flink_venv/bin/pip install apache-flink>=1.19.0 confluent-kafka -c build-constraints.txt

setup-main-venv:
	$(PYENV) install -s 3.11.9
	PYENV_VERSION=3.11.9 $(PYENV) exec python -m venv main_venv
	./main_venv/bin/pip install --upgrade pip
	./main_venv/bin/pip install -r requirements.txt -c build-constraints.txt

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

docker-pipeline:
	docker compose exec maya-app /app/main_venv/bin/python run_pipeline.py $(FLAGS)

# Legacy support / utility targets
PREFIX ?= maya:dashboard
SEED_DIR ?= data/seeds
SECRET_DIR ?= secret_data

redis-publish:
	$(MAIN_PY) -m apps.tools.publish_dashboard_data_to_redis --prefix $(PREFIX)

redis-check:
	$(MAIN_PY) -m apps.tools.check_redis_publish --prefix $(PREFIX)

setup-dev:
	@mkdir -p $(SECRET_DIR)
	@if [ -z "$$(find $(SECRET_DIR) -mindepth 1 -maxdepth 1 -type f | head -n 1)" ]; then \
		echo "[setup-dev] $(SECRET_DIR) is empty. Seeding samples..."; \
		cp $(SEED_DIR)/*.csv $(SECRET_DIR)/; \
	fi

reset-data:
	@mkdir -p $(SECRET_DIR)
	@find $(SECRET_DIR) -mindepth 1 -maxdepth 1 -type f -delete
	@cp $(SEED_DIR)/*.csv $(SECRET_DIR)/
