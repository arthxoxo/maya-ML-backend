#!/bin/bash
set -e

# Determine which component to start
COMPONENT=${1:-dashboard}

case "$COMPONENT" in
  dashboard)
    echo "Starting Maya Dashboard (Python 3.11)..."
    exec /app/main_venv/bin/python -m streamlit run apps/dashboard/streamlit_dashboard.py
    ;;
  flink)
    echo "Starting Sentiment Streaming Job (Python 3.10)..."
    # Ensure Java is reachable (dynamic architecture support)
    export JAVA_HOME=$(find /usr/lib/jvm -maxdepth 1 -name "java-17-active" | head -n 1)
    if [ -z "$JAVA_HOME" ]; then
        export JAVA_HOME=$(find /usr/lib/jvm -maxdepth 1 -name "java-17-openjdk-*" | head -n 1)
    fi
    exec /app/flink_venv/bin/python3 pipelines/streaming/flink_sentiment_job.py
    ;;
  producer)
    echo "Starting Kafka CSV Producer (Python 3.10)..."
    exec /app/flink_venv/bin/python3 pipelines/ingestion/kafka_csv_producer.py --delay 0.1
    ;;
  ingestor)
    echo "Starting Database Ingestor (Python 3.11)..."
    exec /app/main_venv/bin/python pipelines/ingestion/db_ingestor.py
    ;;
  shell)
    exec /bin/bash
    ;;
  *)
    echo "Unknown component: $COMPONENT"
    echo "Usage: ./entrypoint.sh [dashboard|flink|producer|ingestor|shell]"
    exit 1
    ;;
esac
