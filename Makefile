PY ?= ./flink_venv/bin/python
PREFIX ?= maya:dashboard
SEED_DIR ?= data/seeds
SECRET_DIR ?= secret_data

.PHONY: redis-publish redis-check setup-dev reset-data

redis-publish:
	$(PY) -m apps.tools.publish_dashboard_data_to_redis --prefix $(PREFIX)

redis-check:
	$(PY) -m apps.tools.check_redis_publish --prefix $(PREFIX)

setup-dev:
	@mkdir -p $(SECRET_DIR)
	@if [ -z "$$(find $(SECRET_DIR) -mindepth 1 -maxdepth 1 -type f | head -n 1)" ]; then \
		echo "[setup-dev] $(SECRET_DIR) is empty. Seeding schema validation CSVs from $(SEED_DIR)..."; \
		cp $(SEED_DIR)/*.csv $(SECRET_DIR)/; \
		echo "[setup-dev] Seed files copied. NOTE: These are tiny schema-validation samples, not training data."; \
	else \
		echo "[setup-dev] $(SECRET_DIR) already has files. Skipping seed copy to avoid overwriting real data."; \
	fi

reset-data:
	@mkdir -p $(SECRET_DIR)
	@echo "[reset-data] Clearing $(SECRET_DIR) and reseeding from $(SEED_DIR)..."
	@find $(SECRET_DIR) -mindepth 1 -maxdepth 1 -type f -delete
	@cp $(SEED_DIR)/*.csv $(SECRET_DIR)/
	@echo "[reset-data] Done. NOTE: Seed files are for schema validation only."
