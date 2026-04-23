"""
Bulk Sentiment Processor — Step 1 of the Maya ML pipeline.

Reads raw whatsapp_messages.csv from secret_data/, runs the CardiffNLP
RoBERTa model on all user messages, and saves sentiment_scores.csv to
artifacts/sentiment/. This file is then consumed by:
  - feature_engineering.py  (Step 2)
  - build_gnn_nodes_from_flink.py  (Step 3, fallback path)
  - train_xgb_shap_sentiment.py  (Step 5)
"""

import os
import sys
import time
from pathlib import Path

import pandas as pd
from tqdm import tqdm

# Ensure project-root imports resolve when run as a module
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from app_config import RAW_DATA_DIR, SECRET_DATA_DIR, SENTIMENT_ARTIFACT_DIR


def _find_csv(filename: str) -> Path:
    """Try secret_data/, then RAW_DATA_DIR, for both naming conventions."""
    candidates = [
        SECRET_DATA_DIR / filename,
        SECRET_DATA_DIR / f"maya_{filename}",
        RAW_DATA_DIR / filename,
        RAW_DATA_DIR / f"maya_{filename}",
    ]
    for p in candidates:
        if p.exists():
            return p
    raise FileNotFoundError(
        f"Could not find {filename} in secret_data/ or RAW_DATA_DIR. "
        f"Checked: {[str(c) for c in candidates]}"
    )


def main():
    fast_mode = os.getenv("MAYA_PIPELINE_FAST", "0").lower() in ("1", "true", "yes")

    # Map input filenames to their standardized output artifact names
    tasks = [
        ("whatsapp_messages.csv", "sentiment_scores.csv", "message"),
    ]

    pipeline_initialized = False
    pipe = None
    device = None

    for input_file, output_file, text_col in tasks:
        try:
            csv_path = _find_csv(input_file)
        except FileNotFoundError:
            print(f"[Skip] {input_file} not found, skipping sentiment task.")
            continue

        output_path = SENTIMENT_ARTIFACT_DIR / output_file
        output_path.parent.mkdir(parents=True, exist_ok=True)

        if fast_mode and output_path.exists():
            print(f"[Fast Mode] Artifact already exists at {output_path}, skipping {input_file}.")
            continue

        print(f"\n--- Processing {input_file} ---")
        df = pd.read_csv(csv_path)

        # Filter for user messages only if role column exists
        if "role" in df.columns:
            df = df[df["role"].fillna("").str.lower() == "user"].copy()

        if df.empty:
            print(f"No rows to process in {input_file}.")
            continue

        print(f"Processing {len(df):,} items from {input_file}...")

        if fast_mode:
            print("[Fast Mode] Using heuristic sentiment (bypassing Transformer)...")
            import re
            _NEG = {"bad", "worse", "worst", "hate", "angry", "upset", "frustrated", "annoyed",
                    "terrible", "awful", "slow", "broken", "error", "issue", "problem", "failed",
                    "not", "never", "no", "poor", "difficult", "hard", "bug", "crash"}
            _POS = {"good", "great", "awesome", "nice", "love", "happy", "thanks", "thankyou",
                    "resolved", "perfect", "excellent", "fast", "smooth", "best", "cool", "super"}

            def _heuristic(text: str):
                s = str(text or "").strip().lower()
                if not s:
                    return 0.0, 0.5, "neutral"
                tokens = re.findall(r"[a-z']+", s)
                if not tokens:
                    return 0.0, 0.5, "neutral"
                pos = sum(1 for t in tokens if t in _POS)
                neg = sum(1 for t in tokens if t in _NEG)
                raw = float(max(min(((pos - neg) / max(len(tokens), 5)) * 2.5, 1.0), -1.0))
                lbl = "positive" if raw > 0.05 else ("negative" if raw < -0.04 else "neutral")
                return round(raw, 4), round(abs(raw), 4), lbl

            results_list = [_heuristic(t) for t in df[text_col].fillna("").astype(str).tolist()]
            df["sentiment_score"] = [r[0] for r in results_list]
            df["sentiment_confidence"] = [r[1] for r in results_list]
            df["sentiment_label"] = [r[2] for r in results_list]
        else:
            if not pipeline_initialized:
                import torch
                from transformers import pipeline
                from lib.device_utils import resolve_device

                _dev = resolve_device()
                device = str(_dev)

                print("Initializing CardiffNLP RoBERTa model (this will be used for all files)...")
                pipe = pipeline(
                    "sentiment-analysis",
                    model="cardiffnlp/twitter-roberta-base-sentiment-latest",
                    tokenizer="cardiffnlp/twitter-roberta-base-sentiment-latest",
                    device=device,
                )
                pipeline_initialized = True

            batch_size = 32
            texts = df[text_col].fillna("").astype(str).tolist()
            scores = []
            confs = []
            labels = []

            start_time = time.time()
            desc = f"Sentiment ({input_file})"
            for i in tqdm(range(0, len(texts), batch_size), desc=desc, unit="batch"):
                batch = [t[:512] for t in texts[i : i + batch_size]]
                out = pipe(batch, truncation=True, max_length=256)
                for r in out:
                    label = r["label"].lower()
                    score = r["score"]
                    confs.append(round(score, 4))
                    labels.append(label)
                    if "positive" in label or label == "label_2":
                        scores.append(round(score, 4))
                    elif "negative" in label or label == "label_0":
                        scores.append(round(-score, 4))
                    else:
                        scores.append(0.0)

            print(f"Inference complete in {time.time() - start_time:.2f}s")
            df["sentiment_score"] = scores
            df["sentiment_confidence"] = confs
            df["sentiment_label"] = labels

        print(f"Saving {len(df):,} rows to {output_path}...")
        df.to_csv(output_path, index=False)

    print("\n✅ Bulk sentiment processing complete.")


if __name__ == "__main__":
    main()
