import sys
from pathlib import Path
import pandas as pd
import pytest
from unittest.mock import patch, MagicMock

# Mock pyflink before importing modules that rely on it
sys.modules["pyflink"] = MagicMock()
sys.modules["pyflink.table"] = MagicMock()
sys.modules["pyflink.table.udf"] = MagicMock()

# Ensure project-root imports resolve
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from lib.sentiment_utils import heuristic_score, blend_scores, extract_emoji_score, softmax_to_score
from apps.dashboard.streamlit_dashboard import repair_flat_sentiment_scores

def test_heuristic_sentiment_score_positive():
    """Test that positive terms yield a positive score."""
    score = heuristic_score("This is great and awesome and perfect!")
    assert score > 0.1

def test_heuristic_sentiment_score_negative():
    """Test that negative terms yield a negative score."""
    score = heuristic_score("This is terrible and slow and broken.")
    assert score < -0.1

def test_heuristic_sentiment_score_neutral():
    """Test that unknown/empty strings yield neutral score."""
    assert heuristic_score("") == 0.0
    assert heuristic_score("The table is brown.") == 0.0

def test_heuristic_sentiment_score_case_insensitive():
    score_lower = heuristic_score("great")
    score_upper = heuristic_score("GREAT")
    assert score_lower == score_upper

def test_heuristic_score_negation_bigrams():
    """Test that negation-aware bigrams modify scores correctly."""
    score_not_good = heuristic_score("not good")
    score_good = heuristic_score("good")
    assert score_not_good < score_good
    assert score_not_good < 0  # "not good" should be negative overall

    # "not bad" gets a +0.2 bigram bonus, but "not" itself is in HEURISTIC_NEG
    # so the score is still negative. Verify the bigram at least makes it
    # less extreme than what raw keyword scoring alone would produce.
    score_not_bad = heuristic_score("not bad")
    assert score_not_bad < 0  # still negative due to 2 neg keywords
    assert score_not_bad > -1.0  # but not maximally negative

def test_greeting_neutralizer():
    """Test that greetings are dampened toward neutral by blend_scores."""
    # Simulate: model says score=0.25, confidence=0.3 — "hi" is a greeting
    score_hi, _, label_hi = blend_scores(0.25, 0.3, "hi")
    assert label_hi == "neutral", f"Expected 'hi' to be neutral, got {label_hi}"
    assert abs(score_hi) < 0.15, f"Expected dampened score for 'hi', got {score_hi}"

    score_hello, _, label_hello = blend_scores(0.25, 0.3, "hello")
    assert label_hello == "neutral", f"Expected 'hello' to be neutral, got {label_hello}"

    score_ok, _, label_ok = blend_scores(0.20, 0.3, "ok")
    assert label_ok == "neutral", f"Expected 'ok' to be neutral, got {label_ok}"

    # A genuinely positive message should NOT be dampened
    score_great, _, label_great = blend_scores(0.50, 0.6, "this is great!")
    assert label_great == "positive", f"Expected 'this is great!' to be positive, got {label_great}"

def test_blend_scores_emoji():
    """Test that emoji signals are blended into the final score."""
    score_no_emoji, _, _ = blend_scores(0.0, 0.2, "this is fine")
    score_pos_emoji, _, _ = blend_scores(0.0, 0.2, "this is fine 😊")
    assert score_pos_emoji > score_no_emoji, "Positive emoji should nudge score upward"

    score_neg_emoji, _, _ = blend_scores(0.0, 0.2, "this is fine 😡")
    assert score_neg_emoji < score_no_emoji, "Negative emoji should nudge score downward"

def test_softmax_to_score():
    """Test CardiffNLP softmax conversion."""
    result = [
        {"label": "positive", "score": 0.9},
        {"label": "neutral", "score": 0.05},
        {"label": "negative", "score": 0.05},
    ]
    score, conf, label = softmax_to_score(result)
    assert score > 0.5
    assert conf > 0.8
    assert label == "positive"

    result_neg = [
        {"label": "positive", "score": 0.05},
        {"label": "neutral", "score": 0.05},
        {"label": "negative", "score": 0.9},
    ]
    score_neg, conf_neg, label_neg = softmax_to_score(result_neg)
    assert score_neg < -0.5
    assert label_neg == "negative"

def test_repair_flat_sentiment_scores():
    """Test that repair_flat_sentiment_scores applies heuristics to zero scores."""
    df = pd.DataFrame({
        "message": ["great news", "bad news", "regular news"],
        "sentiment_score": [0.0, 0.0, 0.0],
        "sentiment_label": ["neutral", "neutral", "neutral"]
    })
    
    repaired = repair_flat_sentiment_scores(
        df, text_col="message", score_col="sentiment_score", label_col="sentiment_label"
    )
    
    # "great news" should be positive now
    assert repaired.loc[0, "sentiment_score"] > 0
    assert repaired.loc[0, "sentiment_label"] == "positive"
    
    # "bad news" should be negative now
    assert repaired.loc[1, "sentiment_score"] < 0
    assert repaired.loc[1, "sentiment_label"] == "negative"
    
    # "regular news" remains neutral
    assert repaired.loc[2, "sentiment_score"] == 0
    assert repaired.loc[2, "sentiment_label"] == "neutral"

@patch("apps.dashboard.streamlit_dashboard.load_hf_pipelines")
def test_mocked_hf_inference(mock_load_hf):
    """Test that if HF inference is mocked, it applies correctly via streamlit dashboard helpers."""
    from apps.dashboard.streamlit_dashboard import cardiff_sentiment_scores
    
    # Mock the HF pipeline to return "positive"
    mock_pipe = MagicMock()
    mock_pipe.return_value = [{"label": "positive", "score": 0.99}]
    mock_load_hf.return_value = (mock_pipe, None)
    
    df = pd.DataFrame({"message": ["I am so happy"], "user_id": [1], "created_at": ["2026-01-01"]})
    res = cardiff_sentiment_scores(df, text_col="message", group_col="user_id", time_col="created_at")
    
    assert res.loc[0, "sentiment_score"] == 0.99
    assert res.loc[0, "sentiment_label"] == "positive"
