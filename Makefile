PY ?= ./flink_venv/bin/python
PREFIX ?= maya:dashboard

redis-publish:
	$(PY) -m apps.tools.publish_dashboard_data_to_redis --prefix $(PREFIX)

redis-check:
	$(PY) -m apps.tools.check_redis_publish --prefix $(PREFIX)
