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

from pyflink.table import DataTypes, EnvironmentSettings, TableEnvironment, Row
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
    "disappointing", "disappointed", "useless", "boring", "confused", "confusing",
    "difficult", "hard", "stuck", "waiting", "lag", "bug", "crash", "poor",
    "wrong", "miss", "missed", "lost", "waste", "annoying", "painful", "sad",
    "unhappy", "worried", "stress", "stressed", "tired", "sucks", "horrible",
}
_POSITIVE_TERMS = {
    "good", "great", "awesome", "nice", "love", "happy", "thanks", "thankyou",
    "resolved", "perfect", "excellent", "fast", "smooth", "amazing", "wonderful",
    "helpful", "cool", "super", "best", "fantastic", "brilliant", "easy",
    "quick", "convenient", "reliable", "works", "working", "fixed", "solved",
    "appreciate", "glad", "pleased", "thx", "ty", "yay", "wow", "lol", "haha",
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
    # Use smaller denominator for short messages to amplify signal
    denom = max(len(tokens), 4) if len(tokens) <= 8 else max(len(tokens), 6)
    raw = (pos_hits - neg_hits) / denom

    if "!" in s:
        raw *= 1.2
    if "?" in s and neg_hits > 0:
        raw -= 0.05  # Questions with negative terms lean negative
    if any(w in s for w in ["not good", "not happy", "never again", "don't like", "can't"]):
        raw -= 0.2
    if any(w in s for w in ["not bad", "works now", "all good", "thank you", "no problem"]):
        raw += 0.2
    return float(max(min(raw * 2.5, 1.0), -1.0))


def _get_hf_sentiment_pipe():
    global _HF_SENTIMENT_PIPE, _HF_SENTIMENT_UNAVAILABLE
    if _HF_SENTIMENT_PIPE is not None:
        return _HF_SENTIMENT_PIPE
    if _HF_SENTIMENT_UNAVAILABLE:
        return None

    try:
        import sys
        from pathlib import Path as _Path
        sys.path.insert(0, str(_Path(__file__).resolve().parents[2]))
        from lib.device_utils import resolve_device
        from transformers import pipeline

        _dev = resolve_device()
        _HF_SENTIMENT_PIPE = pipeline(
            "sentiment-analysis",
            model="cardiffnlp/twitter-roberta-base-sentiment-latest",
            tokenizer="cardiffnlp/twitter-roberta-base-sentiment-latest",
            device=str(_dev),
        )
    except Exception:
        _HF_SENTIMENT_UNAVAILABLE = True
        _HF_SENTIMENT_PIPE = None
    return _HF_SENTIMENT_PIPE


def _hf_sentiment_full(text: str) -> tuple[float, float]:
    msg = str(text or "").strip()
    if not msg:
        return 0.0, 0.0

    pipe = _get_hf_sentiment_pipe()
    if pipe is None:
        val = _heuristic_sentiment_score(msg)
        return val, abs(val)

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
        return float(max(min(val, 1.0), -1.0)), score
    except Exception:
        val = _heuristic_sentiment_score(msg)
        return val, abs(val)


@udf(result_type=DataTypes.ROW([
    DataTypes.FIELD("score", DataTypes.DOUBLE()),
    DataTypes.FIELD("confidence", DataTypes.DOUBLE())
]))
def compute_sentiment_all(text: str) -> Row:
    score, conf = _hf_sentiment_full(text)
    return Row(round(float(score), 4), round(float(conf), 4))


@udf(result_type=DataTypes.STRING())
def sentiment_label(score: float) -> str:
    if score is None:
        return "neutral"
    if score > 0.05:
        return "positive"
    if score < -0.05:
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

    t_env.create_temporary_function("compute_sentiment_all", compute_sentiment_all)
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
            `sentiment_confidence` DOUBLE,
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
            `sentiment_confidence` DOUBLE,
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

    # Single View enrichment to avoid double-inference across branches
    t_env.execute_sql(
        f"""
        CREATE TEMPORARY VIEW enriched_messages AS
        SELECT
            message_id,
            session_id,
            sender_user_id,
            `role`,
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
            s.score AS sentiment_score,
            s.confidence AS sentiment_confidence,
            sentiment_label(s.score) AS sentiment_label
        FROM (
            SELECT
                id AS message_id,
                session_id,
                sender_user_id,
                `role`,
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
                compute_sentiment_all(message) AS s
            FROM raw_whatsapp_messages
            WHERE LOWER(COALESCE(`role`, '')) = 'user'
        )
        """
    )

    messages_enriched_table = t_env.from_path("enriched_messages")
    console_table = t_env.sql_query(
        """
        SELECT
            message_id,
            session_id,
            `role`,
            sentiment_score,
            sentiment_confidence,
            sentiment_label
        FROM enriched_messages
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
