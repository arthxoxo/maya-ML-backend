from __future__ import annotations

import argparse
import os
import sys


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Check whether dashboard data is published in Redis")
    p.add_argument("--redis-url", type=str, default=os.getenv("REDIS_URL", "").strip())
    p.add_argument("--prefix", type=str, default=os.getenv("MAYA_REDIS_PREFIX", "maya:dashboard").strip() or "maya:dashboard")
    p.add_argument("--min-keys", type=int, default=1, help="Minimum expected key count for prefix")
    p.add_argument("--sample-key", type=str, default="user_embeddings", help="Key suffix to sample for payload length")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    if not args.redis_url:
        raise SystemExit("REDIS_URL is required. Set env var or pass --redis-url.")

    try:
        import redis  # type: ignore
    except Exception as exc:
        raise SystemExit("redis package not installed in this environment. Run: pip install redis") from exc

    client = redis.Redis.from_url(args.redis_url, decode_responses=True)
    client.ping()

    pattern = f"{args.prefix}:*"
    keys = sorted(client.scan_iter(match=pattern))
    key_count = len(keys)

    sample_full_key = f"{args.prefix}:{args.sample_key}"
    sample_payload = client.get(sample_full_key)
    sample_strlen = len(sample_payload) if sample_payload else 0

    print(f"redis_ok=true")
    print(f"prefix={args.prefix}")
    print(f"key_count={key_count}")
    print(f"sample_key={sample_full_key}")
    print(f"sample_strlen={sample_strlen}")

    if keys:
        print("keys_preview=")
        for k in keys[:20]:
            print(k)

    if key_count < int(args.min_keys):
        print(f"ERROR: expected at least {args.min_keys} keys for prefix '{args.prefix}', found {key_count}.")
        raise SystemExit(2)

    if sample_strlen <= 0:
        print(f"ERROR: sample key '{sample_full_key}' is empty or missing.")
        raise SystemExit(3)


if __name__ == "__main__":
    main()
