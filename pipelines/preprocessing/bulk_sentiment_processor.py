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
import re
import sys
import time
from pathlib import Path

import pandas as pd
from tqdm import tqdm

# Ensure project-root imports resolve when run as a module
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from app_config import RAW_DATA_DIR, SECRET_DATA_DIR, SENTIMENT_ARTIFACT_DIR
from lib.online_store import save_artifact_df

_CTX_SEPARATOR = " [CTX] "
_MAX_CONTEXT_CHARS = 512

# ── Emoji sentiment map ────────────────────────────────────────────────────
# RoBERTa's tokenizer strips most emojis.  We extract this signal separately
# and blend it into the final score so WhatsApp-style affective cues survive.
_EMOJI_SENTIMENT: dict[str, float] = {
    # Positive
    "😊": 0.6, "😄": 0.7, "😃": 0.7, "😁": 0.6, "🙂": 0.3, "😀": 0.6,
    "🥰": 0.8, "😍": 0.8, "❤️": 0.7, "💕": 0.6, "💖": 0.7, "🩷": 0.6,
    "👍": 0.5, "👏": 0.5, "🙏": 0.5, "🤗": 0.6, "😘": 0.7, "🎉": 0.7,
    "✨": 0.4, "💯": 0.6, "🔥": 0.4, "😂": 0.4, "🤣": 0.4, "💪": 0.5,
    "🥳": 0.7, "☺️": 0.5, "😇": 0.5, "🫶": 0.6,
    # Negative
    "😢": -0.6, "😭": -0.7, "😞": -0.6, "😔": -0.5, "😟": -0.5,
    "😤": -0.7, "😡": -0.8, "🤬": -0.9, "😠": -0.7, "💔": -0.7,
    "👎": -0.5, "😩": -0.6, "😫": -0.6, "🙁": -0.4, "☹️": -0.5,
    "😰": -0.5, "😥": -0.5, "😓": -0.4, "🤮": -0.7, "🤢": -0.5,
    "😒": -0.4, "🥺": -0.3, "😿": -0.5,
}

_HEURISTIC_NEG = {
    "bad", "worse", "worst", "hate", "angry", "upset", "frustrated", "annoyed",
    "terrible", "awful", "slow", "broken", "error", "issue", "problem", "failed",
    "not", "never", "no", "poor", "difficult", "hard", "bug", "crash",
    "disappointing", "disappointed", "useless", "boring", "confused", "confusing",
    "stuck", "waiting", "lag", "wrong", "miss", "missed", "lost", "waste",
    "annoying", "painful", "sad", "unhappy", "worried", "stress", "stressed",
    "tired", "sucks", "horrible",
}
_HEURISTIC_POS = {
    "good", "great", "awesome", "nice", "love", "happy", "thanks", "thankyou",
    "resolved", "perfect", "excellent", "fast", "smooth", "best", "cool", "super",
    "amazing", "wonderful", "helpful", "fantastic", "brilliant", "easy",
    "quick", "convenient", "reliable", "works", "working", "fixed", "solved",
    "appreciate", "glad", "pleased", "thx", "ty", "yay", "wow", "lol", "haha",
}


def _extract_emoji_score(text: str) -> tuple[float, int]:
    """Return (weighted_emoji_sentiment, emoji_count) from text."""
    total = 0.0
    count = 0
    for ch in text:
        if ch in _EMOJI_SENTIMENT:
            total += _EMOJI_SENTIMENT[ch]
            count += 1
    # Also check multi-char emoji sequences (e.g. ❤️, ☺️)
    for emoji, val in _EMOJI_SENTIMENT.items():
        if len(emoji) > 1 and emoji in text:
            total += val
            count += 1
    if count == 0:
        return 0.0, 0
    return float(max(min(total / count, 1.0), -1.0)), count


def _heuristic_score(text: str) -> float:
    """Keyword-based sentiment score as an ensemble signal."""
    s = str(text or "").strip().lower()
    if not s:
        return 0.0
    tokens = re.findall(r"[a-z']+", s)
    if not tokens:
        return 0.0

    pos = sum(1 for t in tokens if t in _HEURISTIC_POS)
    neg = sum(1 for t in tokens if t in _HEURISTIC_NEG)
    denom = max(len(tokens), 4) if len(tokens) <= 8 else max(len(tokens), 6)
    raw = (pos - neg) / denom

    if "!" in s:
        raw *= 1.2
    if "?" in s and neg > 0:
        raw -= 0.05
    # Negation-aware bigrams
    if any(w in s for w in ["not good", "not happy", "never again", "don't like", "can't"]):
        raw -= 0.2
    if any(w in s for w in ["not bad", "works now", "all good", "thank you", "no problem"]):
        raw += 0.2
    return float(max(min(raw * 2.5, 1.0), -1.0))


def _softmax_to_score(result_list: list[dict]) -> tuple[float, float, str]:
    """Convert top_k=None softmax output to (score, confidence, label).

    CardiffNLP labels: negative / neutral / positive.
    Score  = P(positive) - P(negative)  →  continuous in [-1, +1]
    Confidence = 1 - P(neutral)         →  how "opinionated" the model is
    """
    probs = {"negative": 0.0, "neutral": 0.0, "positive": 0.0}
    for entry in result_list:
        lbl = str(entry.get("label", "")).strip().lower()
        p = float(entry.get("score", 0.0))
        if "positive" in lbl or lbl in {"label_2", "2"}:
            probs["positive"] = p
        elif "negative" in lbl or lbl in {"label_0", "0"}:
            probs["negative"] = p
        else:
            probs["neutral"] = p

    raw_score = probs["positive"] - probs["negative"]
    confidence = 1.0 - probs["neutral"]

    # Dampen score by confidence: when P(neutral) is high, the P(pos)-P(neg)
    # difference is noise, not real sentiment.  Multiplying by confidence
    # pulls uncertain predictions toward zero so greetings like "Hello" and
    # functional messages don't get falsely classified as positive.
    score = raw_score * confidence

    # Derive label from the dampened score
    if score > 0.15:
        label = "positive"
    elif score < -0.15:
        label = "negative"
    else:
        label = "neutral"

    return round(float(score), 4), round(float(max(confidence, 0.01)), 4), label


def _blend_scores(
    model_score: float,
    model_confidence: float,
    text: str,
) -> tuple[float, float, str]:
    """Blend model score with emoji and heuristic signals.

    When model confidence is high (> 0.4), trust the model.
    When low, incorporate heuristic and emoji cues.
    """
    emoji_score, emoji_count = _extract_emoji_score(text)
    heur_score = _heuristic_score(text)

    if model_confidence >= 0.4:
        # High confidence: model dominates, emoji is a small nudge
        if emoji_count > 0:
            final = 0.85 * model_score + 0.15 * emoji_score
        else:
            final = model_score
    else:
        # Low confidence zone: blend all three signals
        if emoji_count > 0:
            final = 0.50 * model_score + 0.25 * heur_score + 0.25 * emoji_score
        else:
            final = 0.60 * model_score + 0.40 * heur_score

    final = float(max(min(final, 1.0), -1.0))
    # Adjust confidence upward if emoji/heuristic agree with model direction
    conf = model_confidence
    if emoji_count > 0 and (emoji_score * model_score > 0):
        conf = min(conf + 0.1, 1.0)
    if heur_score * model_score > 0:
        conf = min(conf + 0.05, 1.0)

    # ── Greeting / phatic neutralizer ──────────────────────────────────────
    # Short messages that are greetings, single-word replies, or non-text
    # content should never be classified as positive/negative regardless of
    # what the model says.  CardiffNLP has a strong positive bias for
    # greetings because Twitter training data associates them with friendly
    # interactions.
    _clean = str(text or "").strip().lower()
    _clean_alpha = re.sub(r"[^a-z]", "", _clean)
    _GREETINGS = {
        "hi", "hii", "hiii", "hey", "hello", "helo", "helo", "yo", "sup",
        "hola", "namaste", "heya", "hiya", "howdy",
        "yes", "no", "ok", "okay", "k", "yep", "yea", "yeah", "nah", "nope",
        "hmm", "hm", "ohh", "oh", "ah", "umm", "um",
        "bye", "goodnight", "gn", "gm", "goodmorning",
    }
    is_greeting = _clean_alpha in _GREETINGS
    is_very_short = len(_clean) <= 20
    has_no_emotional_words = (heur_score == 0.0 and emoji_count == 0)
    # Also catch emails, URLs, and single-word non-emotional tokens
    is_non_text = bool(re.match(r"^[\w.@+\-/:#?=&]+$", _clean))

    if is_very_short and (is_greeting or (has_no_emotional_words and is_non_text)):
        final = final * 0.3  # heavily dampen, don't zero out completely
        final = float(max(min(final, 1.0), -1.0))

    if final > 0.15:
        label = "positive"
    elif final < -0.15:
        label = "negative"
    else:
        label = "neutral"

    return round(final, 4), round(conf, 4), label


def _build_batch_context_column(df: pd.DataFrame, text_col: str) -> pd.Series:
    """Build a context-enriched text column using the previous 2 messages
    within each session, sorted by created_at.

    Batch-mode equivalent of the streaming session buffer in
    ``flink_sentiment_job.py``.  Uses pandas ``groupby`` + ``shift``
    so no mutable state is needed.
    """
    work = df[["session_id", text_col]].copy() if "session_id" in df.columns else df[[text_col]].copy()
    work["_text_clean"] = work[text_col].fillna("").astype(str).str.strip()

    if "session_id" not in work.columns:
        # No session info — fall back to raw text (no context)
        return work["_text_clean"].str[:_MAX_CONTEXT_CHARS]

    work["_prev1"] = work.groupby("session_id")["_text_clean"].shift(1).fillna("")
    work["_prev2"] = work.groupby("session_id")["_text_clean"].shift(2).fillna("")

    def _concat(row):
        parts = [p for p in [row["_prev2"], row["_prev1"]] if p]
        if parts:
            return (_CTX_SEPARATOR.join(parts) + _CTX_SEPARATOR + row["_text_clean"])[:_MAX_CONTEXT_CHARS]
        return row["_text_clean"][:_MAX_CONTEXT_CHARS]

    return work.apply(_concat, axis=1)


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

        # Sort by session + time so context window is chronologically correct
        if "session_id" in df.columns and "created_at" in df.columns:
            df = df.sort_values(["session_id", "created_at"], kind="mergesort").reset_index(drop=True)

        # Build context-enriched text column (previous 2 msgs from same session)
        context_texts = _build_batch_context_column(df, text_col)

        if fast_mode:
            print("[Fast Mode] Using heuristic sentiment (bypassing Transformer)...")
            import re
            _NEG = {"bad", "worse", "worst", "hate", "angry", "upset", "frustrated", "annoyed",
                    "terrible", "awful", "slow", "broken", "error", "issue", "problem", "failed",
                    "not", "never", "no", "poor", "difficult", "hard", "bug", "crash",
                    "disappointing", "disappointed", "useless", "boring", "confused", "confusing",
                    "stuck", "waiting", "lag", "wrong", "miss", "missed", "lost", "waste",
                    "annoying", "painful", "sad", "unhappy", "worried", "stress", "stressed",
                    "tired", "sucks", "horrible"}
            _POS = {"good", "great", "awesome", "nice", "love", "happy", "thanks", "thankyou",
                    "resolved", "perfect", "excellent", "fast", "smooth", "best", "cool", "super",
                    "amazing", "wonderful", "helpful", "fantastic", "brilliant", "easy",
                    "quick", "convenient", "reliable", "works", "working", "fixed", "solved",
                    "appreciate", "glad", "pleased", "thx", "ty", "yay", "wow", "lol", "haha"}

            def _heuristic(text: str):
                s = str(text or "").strip().lower()
                if not s:
                    return 0.0, 0.5, "neutral"
                tokens = re.findall(r"[a-z']+", s)
                if not tokens:
                    return 0.0, 0.5, "neutral"
                pos = sum(1 for t in tokens if t in _POS)
                neg = sum(1 for t in tokens if t in _NEG)
                denom = max(len(tokens), 4) if len(tokens) <= 8 else max(len(tokens), 5)
                raw = float(max(min(((pos - neg) / denom) * 2.5, 1.0), -1.0))
                lbl = "positive" if raw > 0.03 else ("negative" if raw < -0.03 else "neutral")
                return round(raw, 4), round(max(abs(raw), 0.1), 4), lbl

            results_list = [_heuristic(t) for t in context_texts.tolist()]
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

                # Prefer fine-tuned model if available, else fall back to base CardiffNLP
                _finetuned_path = SENTIMENT_ARTIFACT_DIR.parent / "models" / "finetuned_sentiment"
                if _finetuned_path.exists() and (_finetuned_path / "config.json").exists():
                    _model_id = str(_finetuned_path)
                    print(f"Using fine-tuned sentiment model from {_finetuned_path}")
                else:
                    _model_id = "cardiffnlp/twitter-roberta-base-sentiment-latest"
                    print("Using base CardiffNLP RoBERTa model (no fine-tuned model found)")

                pipe = pipeline(
                    "sentiment-analysis",
                    model=_model_id,
                    tokenizer=_model_id,
                    device=device,
                    top_k=None,  # return full softmax distribution
                )
                pipeline_initialized = True

            batch_size = 32
            texts = context_texts.tolist()
            raw_texts = df[text_col].fillna("").astype(str).tolist()  # original text for emoji/heuristic
            scores = []
            confs = []
            labels = []

            start_time = time.time()
            desc = f"Sentiment ({input_file})"
            for i in tqdm(range(0, len(texts), batch_size), desc=desc, unit="batch"):
                batch = [t[:512] for t in texts[i : i + batch_size]]
                batch_raw = raw_texts[i : i + batch_size]
                out = pipe(batch, truncation=True, max_length=256)
                for j, result_list in enumerate(out):
                    # result_list is a list of dicts [{label, score}, ...] for all 3 classes
                    model_score, model_conf, _ = _softmax_to_score(result_list)
                    raw_text = batch_raw[j] if j < len(batch_raw) else ""
                    final_score, final_conf, final_label = _blend_scores(
                        model_score, model_conf, raw_text,
                    )
                    scores.append(final_score)
                    confs.append(final_conf)
                    labels.append(final_label)

            print(f"Inference complete in {time.time() - start_time:.2f}s")
            df["sentiment_score"] = scores
            df["sentiment_confidence"] = confs
            df["sentiment_label"] = labels

        print(f"Saving {len(df):,} rows to {output_path}...")
        save_artifact_df(df, "sentiment_scores", output_path, index=False)

    print("\n✅ Bulk sentiment processing complete.")


if __name__ == "__main__":
    main()
