"""
Kafka CSV Producer — streams local CSV files into Kafka topics as JSON messages.

Each CSV is mapped to its own Kafka topic:
    users.csv             → maya_users
    sessions.csv          → maya_sessions
    feedbacks.csv         → maya_feedbacks
    whatsapp_messages.csv → maya_whatsapp_messages

Modes:
    batch  — Produce all rows from CSV files once (default)
    stream — Continuously replay CSV rows to simulate live stream

Usage:
    python kafka_csv_producer.py                        # produce ALL CSVs (batch)
    python kafka_csv_producer.py --mode stream           # continuous streaming
    python kafka_csv_producer.py --files users sessions  # produce only selected

Requires:
    pip install confluent-kafka
"""

import csv
import json
import time
import argparse
import sys
import os
from pathlib import Path
from confluent_kafka import Producer

from app_config import RAW_DATA_DIR

# ── Configuration ────────────────────────────────────────────────────────────

KAFKA_BROKER = "localhost:9092"
CSV_DIR = RAW_DATA_DIR

# Mapping: csv filename (without extension) → topic name
CSV_TOPIC_MAP = {
    "users":              "maya_users",
    "sessions":           "maya_sessions",
    "feedbacks":          "maya_feedbacks",
    "whatsapp_messages":  "maya_whatsapp_messages",
}

ROW_DELAY_SECONDS = 0.5  # delay between rows to simulate live stream

# ── Helpers ──────────────────────────────────────────────────────────────────

def delivery_callback(err, msg):
    """Called once per produced message to report success or failure."""
    if err is not None:
        print(f"  ✗ Delivery failed: {err}")
    else:
        print(
            f"  ✓ {msg.topic()} | partition {msg.partition()} | "
            f"offset {msg.offset()}"
        )


# Graph metadata for each CSV type (helps downstream GNN graph updates)
GRAPH_METADATA = {
    "users": {
        "node_type": "user",
        "edge_types": ["has_session", "gave_feedback"],
        "id_field": "id",
    },
    "sessions": {
        "node_type": "session",
        "edge_types": ["belongs_to", "contains", "has_feedback"],
        "id_field": "id",
        "parent_field": "user_id",
    },
    "whatsapp_messages": {
        "node_type": "message",
        "edge_types": ["in_session", "followed_by"],
        "id_field": "id",
        "parent_field": "session_id",
    },
    "feedbacks": {
        "node_type": "feedback",
        "edge_types": ["from_user", "about_session"],
        "id_field": "id",
        "parent_field": "user_id",
    },
}


def clean_row(row: dict) -> dict:
    """
    Sanitise a CSV row before serialising to JSON:
    - Strip whitespace from keys and values.
    - Convert empty strings to None (JSON null) so downstream
      consumers don't have to guess.
    - Attempt to coerce obvious numeric values.
    """
    cleaned = {}
    for key, value in row.items():
        key = key.strip()
        if isinstance(value, str):
            value = value.strip()
        if value == "":
            value = None
        else:
            # Try int → float → leave as string
            try:
                value = int(value)
            except (ValueError, TypeError):
                try:
                    value = float(value)
                except (ValueError, TypeError):
                    pass
        cleaned[key] = value
    return cleaned


def produce_csv(producer: Producer, csv_name: str, topic: str,
                delay: float = 0.5, enrich: bool = True) -> int:
    """
    Read a single CSV file and produce every row as a JSON message.

    If enrich=True, adds graph_metadata to each message for downstream
    GNN graph construction.

    Returns the number of rows produced.
    """
    csv_path = CSV_DIR / f"{csv_name}.csv"
    if not csv_path.exists():
        print(f"⚠  File not found, skipping: {csv_path}")
        return 0

    print(f"\n{'─' * 60}")
    print(f"📂  {csv_path.name}  →  topic: {topic}")
    print(f"{'─' * 60}")

    meta = GRAPH_METADATA.get(csv_name, {})

    produced = 0
    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for idx, row in enumerate(reader, start=1):
            payload = clean_row(row)

            # Enrich with graph metadata for GNN downstream
            if enrich and meta:
                payload["_graph_meta"] = {
                    "node_type": meta.get("node_type"),
                    "edge_types": meta.get("edge_types"),
                    "source": csv_name,
                }

            value_bytes = json.dumps(payload, ensure_ascii=False).encode("utf-8")

            # Use the row's "id" as the Kafka key if available, else row index
            key = str(payload.get("id", idx)).encode("utf-8")

            producer.produce(
                topic=topic,
                key=key,
                value=value_bytes,
                callback=delivery_callback,
            )

            # Trigger any queued delivery callbacks (non-blocking)
            producer.poll(0)

            if idx % 100 == 0:
                print(f"  … {idx} rows produced so far")

            time.sleep(delay)
            produced += 1

    # Block until every in-flight message is delivered
    producer.flush()
    print(f"✅  {produced} rows produced to '{topic}'")
    return produced


def stream_continuous(producer: Producer, files: list, delay: float = 1.0):
    """
    Continuously replay CSV data to simulate a live event stream.
    Cycles through files repeatedly until interrupted.
    """
    print("\n" + "═" * 60)
    print("🔁  Starting continuous stream mode...")
    print("    Press Ctrl+C to stop")
    print("═" * 60)

    cycle = 0
    try:
        while True:
            cycle += 1
            print(f"\n🔄  Stream cycle {cycle}")
            for csv_name in files:
                topic = CSV_TOPIC_MAP[csv_name]
                produce_csv(producer, csv_name, topic, delay=delay, enrich=True)
            print(f"    Cycle {cycle} complete. Restarting in 5s...")
            time.sleep(5)
    except KeyboardInterrupt:
        print(f"\n⏹  Stream stopped after {cycle} cycles")


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Stream CSV files into Kafka topics as JSON messages."
    )
    parser.add_argument(
        "--files",
        nargs="+",
        choices=list(CSV_TOPIC_MAP.keys()),
        default=list(CSV_TOPIC_MAP.keys()),
        help="CSV names to produce (default: all).",
    )
    parser.add_argument(
        "--broker",
        default=KAFKA_BROKER,
        help=f"Kafka bootstrap server (default: {KAFKA_BROKER}).",
    )
    parser.add_argument(
        "--delay",
        type=float,
        default=ROW_DELAY_SECONDS,
        help=f"Seconds between rows (default: {ROW_DELAY_SECONDS}).",
    )
    parser.add_argument(
        "--mode",
        choices=["batch", "stream"],
        default="batch",
        help="batch: produce once and exit. stream: continuously replay (default: batch).",
    )
    args = parser.parse_args()

    # ── Create Kafka producer ────────────────────────────────────────
    conf = {
        "bootstrap.servers": args.broker,
        "client.id": "maya-csv-producer",
        "acks": "all",                  # wait for full ISR acknowledgement
        "compression.type": "snappy",   # compress for throughput
        "linger.ms": 5,                 # micro-batch for efficiency
        "batch.size": 32768,
    }
    producer = Producer(conf)
    print(f"🚀  Connected to Kafka @ {args.broker}")
    print(f"    Mode: {args.mode}")

    if args.mode == "stream":
        stream_continuous(producer, args.files, delay=args.delay)
    else:
        total_rows = 0
        start = time.time()

        for csv_name in args.files:
            topic = CSV_TOPIC_MAP[csv_name]
            total_rows += produce_csv(producer, csv_name, topic, delay=args.delay)

        elapsed = time.time() - start
        print(f"\n{'═' * 60}")
        print(f"🏁  Done — {total_rows} total rows in {elapsed:.1f}s")
        print(f"{'═' * 60}")


if __name__ == "__main__":
    main()
