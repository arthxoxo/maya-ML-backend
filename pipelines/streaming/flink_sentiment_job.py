"""
PyFlink Streaming Job — Kafka -> Flink -> Filesystem datasets.

Flow:
    CSV -> Kafka raw topics -> Flink transforms -> filesystem engineered datasets

This job writes Flink-derived datasets used by GNN node building:
    - flink_engineered/users
    - flink_engineered/sessions
    - flink_engineered/feedbacks
    - flink_engineered/messages_sentiment
"""

from __future__ import annotations

import os
from pathlib import Path
import re

from pyflink.table import DataTypes, EnvironmentSettings, TableEnvironment
from pyflink.table.udf import udf

from config import BASE_DIR, FLINK_ENGINEERED_DIR


KAFKA_BROKER = "localhost:9092"
USERS_TOPIC = "maya_users"
SESSIONS_TOPIC = "maya_sessions"
FEEDBACKS_TOPIC = "maya_feedbacks"
WHATSAPP_TOPIC = "maya_whatsapp_messages"
FLINK_OUTPUT_DIR = str(FLINK_ENGINEERED_DIR)
JARS_DIR = str(BASE_DIR / "lib")


def _resolve_java_home() -> str:
    candidates = [
        os.getenv("JAVA_HOME", "").strip(),
        "/opt/homebrew/opt/openjdk@17/libexec/openjdk.jdk/Contents/Home",
        "/opt/homebrew/opt/openjdk@17",
    ]
    for c in candidates:
        if not c:
            continue
        if Path(c, "bin", "java").exists():
            return c
    return ""


_HF_SENTIMENT_PIPE = None
_HF_SENTIMENT_UNAVAILABLE = False
_NEGATIVE_TERMS = {
    "bad", "worse", "worst", "hate", "angry", "upset", "frustrated", "annoyed",
    "terrible", "awful", "slow", "broken", "error", "issue", "problem", "failed",
}
_POSITIVE_TERMS = {
    "good", "great", "awesome", "nice", "love", "happy", "thanks", "thankyou",
    "resolved", "perfect", "excellent", "fast", "smooth",
}


def _heuristic_sentiment_score(text: str) -> float:
    s = str(text or "").strip().lower()
    if not s:
        return 0.0

    tokens = re.findall(r"[a-z']+", s)
    if not tokens:
        return 0.0

    pos_hits = sum(1 for t in tokens if t in _POSITIVE_TERMS)
    neg_hits = sum(1 for t in tokens if t in _NEGATIVE_TERMS)
    raw = (pos_hits - neg_hits) / max(len(tokens), 6)

    if "!" in s:
        raw *= 1.1
    if any(w in s for w in ["not good", "not happy", "never again"]):
        raw -= 0.2
    if any(w in s for w in ["not bad", "works now", "all good"]):
        raw += 0.2
    return float(max(min(raw * 2.0, 1.0), -1.0))


def _get_hf_sentiment_pipe():
    global _HF_SENTIMENT_PIPE, _HF_SENTIMENT_UNAVAILABLE
    if _HF_SENTIMENT_PIPE is not None:
        return _HF_SENTIMENT_PIPE
    if _HF_SENTIMENT_UNAVAILABLE:
        return None

    try:
        from transformers import pipeline

        _HF_SENTIMENT_PIPE = pipeline(
            "sentiment-analysis",
            model="cardiffnlp/twitter-roberta-base-sentiment-latest",
            tokenizer="cardiffnlp/twitter-roberta-base-sentiment-latest",
            device=-1,
        )
    except Exception:
        _HF_SENTIMENT_UNAVAILABLE = True
        _HF_SENTIMENT_PIPE = None
    return _HF_SENTIMENT_PIPE


def _hf_sentiment_score(text: str) -> float:
    msg = str(text or "").strip()
    if not msg:
        return 0.0

    pipe = _get_hf_sentiment_pipe()
    if pipe is None:
        return _heuristic_sentiment_score(msg)

    try:
        out = pipe(msg[:512], truncation=True, max_length=256)
        rec = out[0] if isinstance(out, list) and out else {}
        label = str(rec.get("label", "")).strip().lower()
        score = float(rec.get("score", 0.0))
        if "positive" in label or label in {"label_2", "2"}:
            val = score
        elif "negative" in label or label in {"label_0", "0"}:
            val = -score
        else:
            val = 0.0
        return float(max(min(val, 1.0), -1.0))
    except Exception:
        return _heuristic_sentiment_score(msg)


@udf(result_type=DataTypes.DOUBLE())
def compute_sentiment(text: str) -> float:
    return round(_hf_sentiment_score(text), 4)


@udf(result_type=DataTypes.STRING())
def sentiment_label(score: float) -> str:
    if score is None:
        return "neutral"
    if score > 0.1:
        return "positive"
    if score < -0.1:
        return "negative"
    return "neutral"


def main() -> None:
    java_home = _resolve_java_home()
    if java_home:
        os.environ["JAVA_HOME"] = java_home

    env_settings = EnvironmentSettings.in_streaming_mode()
    t_env = TableEnvironment.create(env_settings)
    t_env.get_config().set("parallelism.default", "1")

    kafka_jar_candidates = list(Path(JARS_DIR).glob("flink-sql-connector-kafka-*.jar"))
    if not kafka_jar_candidates:
        raise FileNotFoundError(
            f"No Kafka connector JAR found in {JARS_DIR}. "
            "Download flink-sql-connector-kafka and place it in ./lib"
        )
    kafka_jar = kafka_jar_candidates[0]
    t_env.get_config().set("pipeline.jars", f"file://{kafka_jar.resolve()}")

    t_env.create_temporary_function("compute_sentiment", compute_sentiment)
    t_env.create_temporary_function("sentiment_label", sentiment_label)

    # Kafka sources
    t_env.execute_sql(
        f"""
        CREATE TABLE raw_users (
            `id`                   INT,
            `created_at`           STRING,
            `updated_at`           STRING,
            `deleted_at`           STRING,
            `first_name`           STRING,
            `last_name`            STRING,
            `timezone`             STRING,
            `country`              STRING,
            `status`               STRING,
            `type`                 STRING,
            `longitude`            DOUBLE,
            `latitude`             DOUBLE,
            `contacts_backfilled`  STRING
        ) WITH (
            'connector'                    = 'kafka',
            'topic'                        = '{USERS_TOPIC}',
            'properties.bootstrap.servers' = '{KAFKA_BROKER}',
            'properties.group.id'          = 'flink-raw-users',
            'scan.startup.mode'            = 'earliest-offset',
            'format'                       = 'json',
            'json.fail-on-missing-field'   = 'false',
            'json.ignore-parse-errors'     = 'true'
        )
        """
    )

    t_env.execute_sql(
        f"""
        CREATE TABLE raw_sessions (
            `id`               INT,
            `user_id`          INT,
            `created_at`       STRING,
            `updated_at`       STRING,
            `deleted_at`       STRING,
            `duration`         DOUBLE,
            `billed_duration`  DOUBLE,
            `transcription`    STRING,
            `summary`          STRING,
            `provider`         STRING
        ) WITH (
            'connector'                    = 'kafka',
            'topic'                        = '{SESSIONS_TOPIC}',
            'properties.bootstrap.servers' = '{KAFKA_BROKER}',
            'properties.group.id'          = 'flink-raw-sessions',
            'scan.startup.mode'            = 'earliest-offset',
            'format'                       = 'json',
            'json.fail-on-missing-field'   = 'false',
            'json.ignore-parse-errors'     = 'true'
        )
        """
    )

    t_env.execute_sql(
        f"""
        CREATE TABLE raw_feedbacks (
            `id`               INT,
            `user_id`          INT,
            `session_id`       INT,
            `message`          STRING,
            `feedback_source`  STRING,
            `created_at`       STRING,
            `updated_at`       STRING,
            `deleted_at`       STRING
        ) WITH (
            'connector'                    = 'kafka',
            'topic'                        = '{FEEDBACKS_TOPIC}',
            'properties.bootstrap.servers' = '{KAFKA_BROKER}',
            'properties.group.id'          = 'flink-raw-feedbacks',
            'scan.startup.mode'            = 'earliest-offset',
            'format'                       = 'json',
            'json.fail-on-missing-field'   = 'false',
            'json.ignore-parse-errors'     = 'true'
        )
        """
    )

    t_env.execute_sql(
        f"""
        CREATE TABLE raw_whatsapp_messages (
            `id`                 INT,
            `session_id`         INT,
            `role`               STRING,
            `message`            STRING,
            `created_at`         STRING,
            `updated_at`         STRING,
            `deleted_at`         STRING,
            `input_tokens`       BIGINT,
            `output_tokens`      BIGINT,
            `model_name`         STRING,
            `cost_usd`           DOUBLE,
            `sender_user_id`     INT,
            `recipient_name`     STRING,
            `status`             STRING
        ) WITH (
            'connector'                    = 'kafka',
            'topic'                        = '{WHATSAPP_TOPIC}',
            'properties.bootstrap.servers' = '{KAFKA_BROKER}',
            'properties.group.id'          = 'flink-raw-whatsapp',
            'scan.startup.mode'            = 'earliest-offset',
            'format'                       = 'json',
            'json.fail-on-missing-field'   = 'false',
            'json.ignore-parse-errors'     = 'true'
        )
        """
    )

    # Filesystem sinks (Flink-derived datasets)
    t_env.execute_sql(
        f"""
        CREATE TABLE users_sink (
            `user_id`              INT,
            `created_at`           STRING,
            `updated_at`           STRING,
            `deleted_at`           STRING,
            `first_name`           STRING,
            `last_name`            STRING,
            `timezone`             STRING,
            `country`              STRING,
            `status`               STRING,
            `type`                 STRING,
            `longitude`            DOUBLE,
            `latitude`             DOUBLE,
            `contacts_backfilled`  STRING
        ) WITH (
            'connector' = 'filesystem',
            'path'      = '{FLINK_OUTPUT_DIR}/users',
            'format'    = 'csv'
        )
        """
    )

    t_env.execute_sql(
        f"""
        CREATE TABLE sessions_sink (
            `session_id`       INT,
            `user_id`          INT,
            `created_at`       STRING,
            `updated_at`       STRING,
            `deleted_at`       STRING,
            `duration`         DOUBLE,
            `billed_duration`  DOUBLE,
            `transcription`    STRING,
            `summary`          STRING,
            `provider`         STRING
        ) WITH (
            'connector' = 'filesystem',
            'path'      = '{FLINK_OUTPUT_DIR}/sessions',
            'format'    = 'csv'
        )
        """
    )

    t_env.execute_sql(
        f"""
        CREATE TABLE feedbacks_sink (
            `feedback_id`       INT,
            `user_id`           INT,
            `session_id`        INT,
            `message`           STRING,
            `feedback_source`   STRING,
            `created_at`        STRING,
            `updated_at`        STRING,
            `deleted_at`        STRING
        ) WITH (
            'connector' = 'filesystem',
            'path'      = '{FLINK_OUTPUT_DIR}/feedbacks',
            'format'    = 'csv'
        )
        """
    )

    t_env.execute_sql(
        f"""
        CREATE TABLE messages_sentiment_sink (
            `message_id`          INT,
            `session_id`          INT,
            `sender_user_id`      INT,
            `role`                STRING,
            `message`             STRING,
            `created_at`          STRING,
            `updated_at`          STRING,
            `deleted_at`          STRING,
            `input_tokens`        BIGINT,
            `output_tokens`       BIGINT,
            `model_name`          STRING,
            `cost_usd`            DOUBLE,
            `recipient_name`      STRING,
            `status`              STRING,
            `sentiment_score`     DOUBLE,
            `sentiment_label`     STRING
        ) WITH (
            'connector' = 'filesystem',
            'path'      = '{FLINK_OUTPUT_DIR}/messages_sentiment',
            'format'    = 'csv'
        )
        """
    )

    t_env.execute_sql(
        """
        CREATE TABLE console_output (
            `message_id`      INT,
            `session_id`      INT,
            `role`            STRING,
            `sentiment_score` DOUBLE,
            `sentiment_label` STRING
        ) WITH ('connector' = 'print')
        """
    )

    users_table = t_env.sql_query(
        """
        SELECT
            id AS user_id,
            created_at,
            updated_at,
            deleted_at,
            first_name,
            last_name,
            timezone,
            country,
            status,
            type,
            longitude,
            latitude,
            contacts_backfilled
        FROM raw_users
        """
    )

    sessions_table = t_env.sql_query(
        """
        SELECT
            id AS session_id,
            user_id,
            created_at,
            updated_at,
            deleted_at,
            duration,
            billed_duration,
            transcription,
            summary,
            provider
        FROM raw_sessions
        """
    )

    feedbacks_table = t_env.sql_query(
        """
        SELECT
            id AS feedback_id,
            user_id,
            session_id,
            message,
            feedback_source,
            created_at,
            updated_at,
            deleted_at
        FROM raw_feedbacks
        """
    )

    messages_enriched_table = t_env.sql_query(
        """
        SELECT
            message_id,
            session_id,
            sender_user_id,
            role,
            message,
            created_at,
            updated_at,
            deleted_at,
            input_tokens,
            output_tokens,
            model_name,
            cost_usd,
            recipient_name,
            status,
            sentiment_score,
            sentiment_label(sentiment_score) AS sentiment_label
        FROM (
            SELECT
                id AS message_id,
                session_id,
                sender_user_id,
                role,
                message,
                created_at,
                updated_at,
                deleted_at,
                input_tokens,
                output_tokens,
                model_name,
                cost_usd,
                recipient_name,
                status,
                compute_sentiment(message) AS sentiment_score
            FROM raw_whatsapp_messages
            WHERE LOWER(COALESCE(role, '')) = 'user'
        ) m
        """
    )

    console_table = t_env.sql_query(
        """
        SELECT
            message_id,
            session_id,
            role,
            sentiment_score,
            sentiment_label(sentiment_score) AS sentiment_label
        FROM (
            SELECT
                id AS message_id,
                session_id,
                role,
                compute_sentiment(message) AS sentiment_score
            FROM raw_whatsapp_messages
            WHERE LOWER(COALESCE(role, '')) = 'user'
        ) c
        """
    )

    stmt = t_env.create_statement_set()
    stmt.add_insert("users_sink", users_table)
    stmt.add_insert("sessions_sink", sessions_table)
    stmt.add_insert("feedbacks_sink", feedbacks_table)
    stmt.add_insert("messages_sentiment_sink", messages_enriched_table)
    stmt.add_insert("console_output", console_table)

    print("\n" + "=" * 64)
    print("Starting Flink pipeline: Kafka raw topics -> flink_engineered/*")
    print("Press Ctrl+C to stop")
    print("=" * 64 + "\n")

    stmt.execute().wait()


if __name__ == "__main__":
    main()
