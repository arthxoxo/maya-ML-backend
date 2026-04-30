"""
Shared Sentiment Scoring Utilities — Single source of truth.

This module centralises the emoji map, heuristic word lists, emoji
extraction, keyword-based scoring, CardiffNLP softmax conversion, and
the ensemble blending logic (including the greeting / phatic neutralizer)
so that **both** the batch pipeline (``bulk_sentiment_processor.py``) and
the streaming pipeline (``flink_sentiment_job.py``) produce identical
sentiment scores.

All functions are **pure** — no model loading, no side effects — so they
are safe to import from any context (batch, Flink UDF, tests, dashboard).
"""

from __future__ import annotations

import re

# ── Emoji sentiment map ────────────────────────────────────────────────────
# RoBERTa's tokenizer strips most emojis.  We extract this signal separately
# and blend it into the final score so WhatsApp-style affective cues survive.
EMOJI_SENTIMENT: dict[str, float] = {
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

# ── Heuristic word lists ──────────────────────────────────────────────────
HEURISTIC_NEG: set[str] = {
    "bad", "worse", "worst", "hate", "angry", "upset", "frustrated", "annoyed",
    "terrible", "awful", "slow", "broken", "error", "issue", "problem", "failed",
    "not", "never", "no", "poor", "difficult", "hard", "bug", "crash",
    "disappointing", "disappointed", "useless", "boring", "confused", "confusing",
    "stuck", "waiting", "lag", "wrong", "miss", "missed", "lost", "waste",
    "annoying", "painful", "sad", "unhappy", "worried", "stress", "stressed",
    "tired", "sucks", "horrible",
}
HEURISTIC_POS: set[str] = {
    "good", "great", "awesome", "nice", "love", "happy", "thanks", "thankyou",
    "resolved", "perfect", "excellent", "fast", "smooth", "best", "cool", "super",
    "amazing", "wonderful", "helpful", "fantastic", "brilliant", "easy",
    "quick", "convenient", "reliable", "works", "working", "fixed", "solved",
    "appreciate", "glad", "pleased", "thx", "ty", "yay", "wow", "lol", "haha",
}

# ── Greeting / phatic tokens ─────────────────────────────────────────────
# Short messages matching these should never be strongly positive/negative.
GREETINGS: set[str] = {
    "hi", "hii", "hiii", "hey", "hello", "helo", "yo", "sup",
    "hola", "namaste", "heya", "hiya", "howdy",
    "yes", "no", "ok", "okay", "k", "yep", "yea", "yeah", "nah", "nope",
    "hmm", "hm", "ohh", "oh", "ah", "umm", "um",
    "bye", "goodnight", "gn", "gm", "goodmorning",
}
# ── Command verbs (Neutralizer) ───────────────────────────────────────────
COMMAND_VERBS: set[str] = {
    "delete", "remove", "cancel", "clear", "stop", "reset",
    "discard", "undo", "erase", "omit", "exclude",
    "update", "edit", "modify", "set", "add", "create", "show", "get", "list",
    "check", "test", "verify", "run", "execute", "filter", "drop", "configure",
}

# ── Technical Keywords (Neutralizer) ──────────────────────────────────────
TECHNICAL_KEYWORDS: set[str] = {
    "config", "configuration", "filter", "logs", "spam", "system", "settings",
    "profile", "account", "data", "database", "server", "connection", "api",
    "token", "subscription", "lead", "leads", "automation", "workflow", "contact", "contacts", "calendars", "meeting",
}

def extract_emoji_score(text: str) -> tuple[float, int]:
    """Return (weighted_emoji_sentiment, emoji_count) from *text*."""
    total = 0.0
    count = 0
    for ch in text:
        if ch in EMOJI_SENTIMENT:
            total += EMOJI_SENTIMENT[ch]
            count += 1
    # Also check multi-char emoji sequences (e.g. ❤️, ☺️)
    for emoji, val in EMOJI_SENTIMENT.items():
        if len(emoji) > 1 and emoji in text:
            total += val
            count += 1
    if count == 0:
        return 0.0, 0
    return float(max(min(total / count, 1.0), -1.0)), count


def heuristic_score(text: str) -> float:
    """Keyword-based sentiment score as an ensemble signal."""
    s = str(text or "").strip().lower()
    if not s:
        return 0.0
    tokens = re.findall(r"[a-z']+", s)
    if not tokens:
        return 0.0

    pos = sum(1 for t in tokens if t in HEURISTIC_POS)
    neg = sum(1 for t in tokens if t in HEURISTIC_NEG)
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


def softmax_to_score(result_list: list[dict]) -> tuple[float, float, str]:
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


def blend_scores(
    model_score: float,
    model_confidence: float,
    text: str,
) -> tuple[float, float, str]:
    """Blend model score with emoji and heuristic signals.

    When model confidence is high (> 0.4), trust the model.
    When low, incorporate heuristic and emoji cues.

    Includes greeting / phatic neutralizer so short greetings
    never score as strongly positive or negative.
    """
    emoji_score, emoji_count = extract_emoji_score(text)
    heur_score = heuristic_score(text)

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

    _clean = str(text or "").strip().lower()
    tokens = [t.strip(".,!?\"'") for t in _clean.split()]
    
    is_greeting = _clean in GREETINGS
    is_very_short = len(tokens) <= 3
    has_no_emotional_words = (heur_score == 0.0 and emoji_count == 0)
    # Also catch emails, URLs, and single-word non-emotional tokens
    is_non_text = bool(re.match(r"^[\w.@+\-/:#?=&]+$", _clean))
    
    # Apply strict neutralizer override for technical commands
    command_in_prefix = any(t in COMMAND_VERBS for t in tokens[:3])
    has_tech_keywords = any(t in TECHNICAL_KEYWORDS for t in tokens)
    is_technical_command = (command_in_prefix or has_tech_keywords) and has_no_emotional_words

    if (is_very_short and (is_greeting or (has_no_emotional_words and is_non_text))) or is_technical_command:
        final = 0.0
        conf = 0.99

    if final > 0.15:
        label = "positive"
    elif final < -0.15:
        label = "negative"
    else:
        label = "neutral"

    return round(final, 4), round(conf, 4), label
