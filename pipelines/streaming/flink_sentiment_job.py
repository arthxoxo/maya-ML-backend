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

from lib.sentiment_utils import (
    extract_emoji_score,
    heuristic_score,
    softmax_to_score,
    blend_scores,
)

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

# Session-scoped message buffer for context-aware sentiment inference.
# Keyed by session_id, each value is a list of the last 2 user messages.
# Module-level state is used because Flink Python UDFs execute in a single
# worker process — this buffer persists across calls within the same job
# execution. It resets if the Flink worker restarts (acceptable trade-off
# for streaming; batch mode in bulk_sentiment_processor.py uses pandas
# shift instead).
_SESSION_MSG_BUFFER: dict[int, list[str]] = {}
_CTX_SEPARATOR = " [CTX] "
_MAX_CONTEXT_CHARS = 512


def _build_context_string(current_msg: str, session_id: int | None) -> str:
    """Prepend prior messages from the same session using [CTX] separator.

    Returns a single string truncated to _MAX_CONTEXT_CHARS that the model
    will score.  The buffer is *read* here but not mutated — the caller is
    responsible for updating it after scoring.
    """
    msg = str(current_msg or "").strip()
    if not msg:
        return ""
    if session_id is None:
        return msg[:_MAX_CONTEXT_CHARS]

    prior = _SESSION_MSG_BUFFER.get(int(session_id), [])
    if prior:
        context = _CTX_SEPARATOR.join(prior) + _CTX_SEPARATOR + msg
    else:
        context = msg
    return context[:_MAX_CONTEXT_CHARS]


def _update_session_buffer(session_id: int | None, message: str) -> None:
    """Append *message* to the session buffer, keeping only the last 2."""
    if session_id is None:
        return
    sid = int(session_id)
    buf = _SESSION_MSG_BUFFER.setdefault(sid, [])
    buf.append(str(message or "").strip())
    if len(buf) > 2:
        _SESSION_MSG_BUFFER[sid] = buf[-2:]


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

        # Prefer fine-tuned model if available, else fall back to base CardiffNLP
        from app_config import ARTIFACTS_DIR as _ARTIFACTS_DIR
        _finetuned_path = _ARTIFACTS_DIR / "models" / "finetuned_sentiment"
        if _finetuned_path.exists() and (_finetuned_path / "config.json").exists():
            _model_id = str(_finetuned_path)
        else:
            _model_id = "cardiffnlp/twitter-roberta-base-sentiment-latest"

        _HF_SENTIMENT_PIPE = pipeline(
            "sentiment-analysis",
            model=_model_id,
            tokenizer=_model_id,
            device=str(_dev),
            top_k=None,  # return full softmax distribution
        )
    except Exception:
        _HF_SENTIMENT_UNAVAILABLE = True
        _HF_SENTIMENT_PIPE = None
    return _HF_SENTIMENT_PIPE


def _hf_sentiment_full(context_text: str, raw_text: str = "") -> tuple[float, float]:
    """Score a (possibly context-enriched) string with ensemble blending.

    *context_text* is the result of ``_build_context_string`` — it may
    contain ``[CTX]`` separators with prior messages prepended.  The
    string is already truncated to ``_MAX_CONTEXT_CHARS``.

    *raw_text* is the original message (before context prepend) used for
    emoji and heuristic extraction.

    Uses ``blend_scores()`` from ``lib.sentiment_utils`` which includes
    the greeting / phatic neutralizer, ensuring parity with the batch
    pipeline in ``bulk_sentiment_processor.py``.
    """
    msg = str(context_text or "").strip()
    if not msg:
        return 0.0, 0.0

    pipe = _get_hf_sentiment_pipe()
    if pipe is None:
        # Fallback: heuristic + emoji blend
        val = heuristic_score(msg)
        emoji_s, emoji_n = extract_emoji_score(raw_text or msg)
        if emoji_n > 0:
            val = 0.6 * val + 0.4 * emoji_s
        return float(max(min(val, 1.0), -1.0)), abs(val)

    try:
        out = pipe(msg[:512], truncation=True, max_length=256)
        # top_k=None returns list of dicts for all 3 classes
        result_list = out if isinstance(out, list) and out and isinstance(out[0], dict) and "label" in out[0] else []
        # Handle nested list from batch mode (shouldn't happen for single input)
        if result_list and isinstance(result_list[0], list):
            result_list = result_list[0]

        model_score, model_conf, _ = softmax_to_score(result_list)

        # Use the shared blend_scores which includes the greeting neutralizer
        final_score, final_conf, _ = blend_scores(
            model_score, model_conf, raw_text or msg,
        )

        return final_score, final_conf
    except Exception:
        val = heuristic_score(msg)
        emoji_s, emoji_n = extract_emoji_score(raw_text or msg)
        if emoji_n > 0:
            val = 0.6 * val + 0.4 * emoji_s
        return float(max(min(val, 1.0), -1.0)), abs(val)


@udf(result_type=DataTypes.ROW([
    DataTypes.FIELD("score", DataTypes.DOUBLE()),
    DataTypes.FIELD("confidence", DataTypes.DOUBLE())
]))
def compute_sentiment_all(text: str, session_id: int) -> Row:
    """Score a message with conversational context from the same session."""
    raw_text = str(text or "").strip()
    context_text = _build_context_string(text, session_id)
    score, conf = _hf_sentiment_full(context_text, raw_text=raw_text)
    _update_session_buffer(session_id, text)
    return Row(round(float(score), 4), round(float(conf), 4))


@udf(result_type=DataTypes.STRING())
def sentiment_label(score: float) -> str:
    if score is None:
        return "neutral"
    if score > 0.15:
        return "positive"
    if score < -0.15:
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
                compute_sentiment_all(message, session_id) AS s
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
