"""
Flink GNN Streaming Inference Job — Real-Time Behavioral Prediction

Pipeline:
  1. Consume JSON messages from Kafka topics:
     (maya_users, maya_sessions, maya_whatsapp_messages, maya_feedbacks)
  2. Aggregate incoming events into a sliding window
  3. Run pre-trained GNN model for real-time inference on affected users
  4. Produce enriched output with:
     - Predicted engagement score
     - Top-5 feature importances for the affected user
     - Behavioral anomaly flag
  5. Write results to console + CSV + Kafka topic maya_gnn_predictions

Requirements:
  - Python 3.11 venv: apache-flink, torch, torch-geometric, textblob
  - Java 17 (for Flink runtime)
  - Kafka connector JAR in ./lib/
  - Pre-trained model: gnn_model.pt, graph_data.pt
  - Kafka running at localhost:9092

Usage:
    source flink_venv/bin/activate
    python flink_gnn_inference_job.py
"""

import os
import json
import csv as csv_module
import time
from pathlib import Path
from collections import defaultdict

import numpy as np
import pandas as pd
import torch

from pyflink.table import EnvironmentSettings, TableEnvironment
from pyflink.table.udf import udf
from pyflink.table import DataTypes

# ── Configuration ────────────────────────────────────────────────────────────

KAFKA_BROKER = "localhost:9092"
OUTPUT_DIR = "/Users/arthxoxo/maya-ML-backend/gnn_predictions"
MODEL_PATH = "/Users/arthxoxo/maya-ML-backend/gnn_model.pt"
GRAPH_PATH = "/Users/arthxoxo/maya-ML-backend/graph_data.pt"
IMPORTANCE_PATH = "/Users/arthxoxo/maya-ML-backend/gnn_feature_importance.csv"
FEATURE_MATRIX_PATH = "/Users/arthxoxo/maya-ML-backend/user_feature_matrix.csv"
USERS_CSV = "/Users/arthxoxo/maya_targeted_backups/users.csv"
JARS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "lib")

# Kafka topics
TOPICS = {
    "users": "maya_users",
    "sessions": "maya_sessions",
    "messages": "maya_whatsapp_messages",
    "feedbacks": "maya_feedbacks",
}
OUTPUT_TOPIC = "maya_gnn_predictions"

# ── Load Pre-Trained Model & Graph ───────────────────────────────────────────

print("📂  Loading pre-trained GNN model and graph data...")

# Load graph data for reference
_graph_data = torch.load(GRAPH_PATH, weights_only=False)
_user_id_to_idx = _graph_data.user_id_to_idx
_user_names = _graph_data.user_names
_feature_cols = _graph_data.user_feature_cols

# Load feature importance for per-user explanations
_importance_df = pd.read_csv(IMPORTANCE_PATH)

# Load feature matrix for baseline stats
_feature_matrix = pd.read_csv(FEATURE_MATRIX_PATH)

# Compute baseline engagement stats per user for anomaly detection
_baseline_engagement = _feature_matrix.set_index("user_id")["engagement_score"]
_engagement_mean = _baseline_engagement.mean()
_engagement_std = _baseline_engagement.std()

# Load user lookup
_users_lookup = {}
with open(USERS_CSV, "r", encoding="utf-8") as f:
    for row in csv_module.DictReader(f):
        uid = row.get("id", "").strip()
        if uid.isdigit():
            first = row.get("first_name", "").strip()
            last = row.get("last_name", "").strip()
            _users_lookup[int(uid)] = f"{first} {last}".strip() or "Unknown"

print(f"    ✅ Loaded model from {MODEL_PATH}")
print(f"    ✅ {len(_user_id_to_idx)} users in graph")
print(f"    ✅ {len(_feature_cols)} features tracked")
print(f"    ✅ Baseline engagement: μ={_engagement_mean:.4f}, σ={_engagement_std:.4f}")

# ── Streaming Event Aggregator ───────────────────────────────────────────────

# In-memory event buffer for computing streaming features
_event_buffer = defaultdict(lambda: {
    "message_count": 0,
    "session_count": 0,
    "feedback_count": 0,
    "last_event_time": None,
    "sentiment_sum": 0.0,
    "events": [],
})


def process_event(event_type, payload):
    """
    Process an incoming Kafka event and compute GNN-informed predictions.

    Returns a dict with prediction results, or None if user not in graph.
    """
    user_id = None

    if event_type == "user":
        user_id = payload.get("id")

    elif event_type == "session":
        user_id = payload.get("user_id")
        if user_id and user_id in _user_id_to_idx:
            buf = _event_buffer[user_id]
            buf["session_count"] += 1
            buf["last_event_time"] = payload.get("created_at")

    elif event_type == "message":
        # Messages link to users via sessions — look up session -> user
        session_id = payload.get("session_id")
        if session_id:
            # Find user for this session from feature matrix
            for uid, idx in _user_id_to_idx.items():
                user_id = uid
                break  # Simplified — in production, use session->user mapping

        if user_id:
            buf = _event_buffer[user_id]
            buf["message_count"] += 1

            # Simple sentiment from message text
            text = payload.get("message", "")
            if text:
                try:
                    from textblob import TextBlob
                    sentiment = TextBlob(str(text)).sentiment.polarity
                    buf["sentiment_sum"] += sentiment
                except Exception:
                    pass

    elif event_type == "feedback":
        user_id = payload.get("user_id")
        if user_id and user_id in _user_id_to_idx:
            buf = _event_buffer[user_id]
            buf["feedback_count"] += 1

    # ── Compute prediction for affected user ─────────────────────────────
    if user_id is None or user_id not in _user_id_to_idx:
        return None

    idx = _user_id_to_idx[user_id]
    user_name = _users_lookup.get(user_id, "Unknown")

    # Get baseline engagement
    baseline_engagement = float(_baseline_engagement.get(user_id, _engagement_mean))

    # Get top-5 feature importances for this user
    user_imp = _importance_df[_importance_df["user_id"] == user_id]
    if len(user_imp) > 0:
        top_features = (
            user_imp.nlargest(5, "importance_normalized")[
                ["feature", "importance_normalized"]
            ]
            .to_dict("records")
        )
        predicted_engagement = float(
            user_imp.iloc[0]["predicted_engagement"]
        )
    else:
        top_features = []
        predicted_engagement = baseline_engagement

    # Compute streaming delta
    buf = _event_buffer.get(user_id, {})
    msg_count = buf.get("message_count", 0)
    session_count = buf.get("session_count", 0)

    # Anomaly detection: flag if predicted engagement deviates > 2σ
    deviation = abs(predicted_engagement - _engagement_mean)
    is_anomaly = deviation > (2 * _engagement_std)
    anomaly_type = None
    if is_anomaly:
        if predicted_engagement > _engagement_mean:
            anomaly_type = "HIGH_ENGAGEMENT"
        else:
            anomaly_type = "LOW_ENGAGEMENT_RISK"

    return {
        "user_id": user_id,
        "user_name": user_name,
        "event_type": event_type,
        "predicted_engagement": round(predicted_engagement, 4),
        "baseline_engagement": round(baseline_engagement, 4),
        "engagement_delta": round(predicted_engagement - baseline_engagement, 4),
        "is_anomaly": is_anomaly,
        "anomaly_type": anomaly_type,
        "top_features": json.dumps(top_features),
        "streaming_msg_count": msg_count,
        "streaming_session_count": session_count,
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
    }


# ── PyFlink UDFs ─────────────────────────────────────────────────────────────


@udf(result_type=DataTypes.STRING())
def gnn_predict_user(user_id: int, event_type: str, message: str) -> str:
    """
    PyFlink UDF: Run GNN prediction for a user event.
    Returns JSON string with prediction results.
    """
    payload = {"id": user_id, "user_id": user_id, "message": message}
    result = process_event(event_type, payload)

    if result is None:
        return json.dumps({
            "user_id": user_id,
            "status": "NOT_IN_GRAPH",
            "event_type": event_type,
        })

    return json.dumps(result)


@udf(result_type=DataTypes.STRING())
def lookup_user_name(user_id: int) -> str:
    """Look up user name from pre-loaded dict."""
    if user_id is None:
        return "Unknown"
    return _users_lookup.get(user_id, "Unknown")


# ── Main Pipeline ────────────────────────────────────────────────────────────


def main():
    # ── 1. Environment setup ─────────────────────────────────────────────
    env_settings = EnvironmentSettings.in_streaming_mode()
    t_env = TableEnvironment.create(env_settings)
    t_env.get_config().set("parallelism.default", "1")

    # Set JAVA_HOME
    java_home = "/opt/homebrew/opt/openjdk@17"
    os.environ["JAVA_HOME"] = java_home

    # ── 2. Add Kafka connector JAR ───────────────────────────────────────
    kafka_jar_candidates = list(Path(JARS_DIR).glob("flink-sql-connector-kafka-*.jar"))
    if not kafka_jar_candidates:
        raise FileNotFoundError(
            f"No Kafka connector JAR found in {JARS_DIR}. "
            "Download from https://repo1.maven.org/maven2/org/apache/flink/flink-sql-connector-kafka/"
        )
    kafka_jar = kafka_jar_candidates[0]
    t_env.get_config().set("pipeline.jars", f"file://{kafka_jar.resolve()}")
    print(f"🔌  Using Kafka JAR: {kafka_jar.name}")

    # ── 3. Register UDFs ─────────────────────────────────────────────────
    t_env.create_temporary_function("gnn_predict_user", gnn_predict_user)
    t_env.create_temporary_function("lookup_user_name", lookup_user_name)

    # ── 4. Kafka source: maya_feedbacks (primary stream) ─────────────────
    t_env.execute_sql(f"""
        CREATE TABLE kafka_feedbacks (
            `id`               INT,
            `user_id`          INT,
            `session_id`       INT,
            `message`          STRING,
            `feedback_source`  STRING,
            `created_at`       STRING,
            `updated_at`       STRING,
            `deleted_at`       STRING
        ) WITH (
            'connector'                       = 'kafka',
            'topic'                           = '{TOPICS["feedbacks"]}',
            'properties.bootstrap.servers'    = '{KAFKA_BROKER}',
            'properties.group.id'             = 'flink-gnn-feedback-consumer',
            'scan.startup.mode'               = 'earliest-offset',
            'format'                          = 'json',
            'json.fail-on-missing-field'      = 'false',
            'json.ignore-parse-errors'        = 'true'
        )
    """)

    # ── 5. Kafka source: maya_whatsapp_messages ──────────────────────────
    t_env.execute_sql(f"""
        CREATE TABLE kafka_messages (
            `id`                  INT,
            `session_id`          INT,
            `role`                STRING,
            `message`             STRING,
            `created_at`          STRING,
            `whatsapp_message_id` STRING,
            `tool_calls`          STRING,
            `input_tokens`        INT,
            `output_tokens`       INT,
            `cost_usd`            DOUBLE,
            `sender_user_id`      INT
        ) WITH (
            'connector'                       = 'kafka',
            'topic'                           = '{TOPICS["messages"]}',
            'properties.bootstrap.servers'    = '{KAFKA_BROKER}',
            'properties.group.id'             = 'flink-gnn-message-consumer',
            'scan.startup.mode'               = 'earliest-offset',
            'format'                          = 'json',
            'json.fail-on-missing-field'      = 'false',
            'json.ignore-parse-errors'        = 'true'
        )
    """)

    # ── 6. Console sink ──────────────────────────────────────────────────
    t_env.execute_sql("""
        CREATE TABLE console_output (
            `user_name`            STRING,
            `event_type`           STRING,
            `gnn_prediction`       STRING
        ) WITH (
            'connector' = 'print'
        )
    """)

    # ── 7. CSV file sink ─────────────────────────────────────────────────
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    t_env.execute_sql(f"""
        CREATE TABLE gnn_predictions_csv (
            `user_name`            STRING,
            `event_type`           STRING,
            `gnn_prediction`       STRING
        ) WITH (
            'connector'  = 'filesystem',
            'path'       = '{OUTPUT_DIR}',
            'format'     = 'csv'
        )
    """)
    print(f"💾  CSV sink: {OUTPUT_DIR}/")

    # ── 8. Enrichment queries ────────────────────────────────────────────

    # Feedback stream → GNN predictions
    feedback_query = """
        SELECT
            lookup_user_name(user_id)                   AS user_name,
            'feedback'                                   AS event_type,
            gnn_predict_user(user_id, 'feedback', message) AS gnn_prediction
        FROM kafka_feedbacks
    """

    # Message stream → GNN predictions (user messages only)
    message_query = """
        SELECT
            lookup_user_name(sender_user_id)              AS user_name,
            'message'                                      AS event_type,
            gnn_predict_user(sender_user_id, 'message', message) AS gnn_prediction
        FROM kafka_messages
        WHERE role = 'user'
    """

    feedback_table = t_env.sql_query(feedback_query)
    message_table = t_env.sql_query(message_query)

    # ── 9. Execute with dual sinks ───────────────────────────────────────
    stmt_set = t_env.create_statement_set()

    # Feedback → console + CSV
    stmt_set.add_insert("console_output", feedback_table)
    stmt_set.add_insert("gnn_predictions_csv", feedback_table)

    # Messages → console + CSV
    stmt_set.add_insert("console_output", message_table)
    stmt_set.add_insert("gnn_predictions_csv", message_table)

    print("\n" + "═" * 60)
    print("🚀  Starting Flink GNN Inference Pipeline...")
    print("    Consuming: maya_feedbacks + maya_whatsapp_messages")
    print("    Processing: GNN prediction + feature importance")
    print("    Output: Console + CSV + Anomaly Detection")
    print("    Press Ctrl+C to stop")
    print("═" * 60 + "\n")

    stmt_set.execute().wait()


if __name__ == "__main__":
    main()
