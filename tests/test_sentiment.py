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

from pipelines.streaming.flink_sentiment_job import _heuristic_sentiment_score
from apps.dashboard.streamlit_dashboard import repair_flat_sentiment_scores

def test_heuristic_sentiment_score_positive():
    """Test that positive terms yield a positive score."""
    score = _heuristic_sentiment_score("This is great and awesome and perfect!")
    assert score > 0.1

def test_heuristic_sentiment_score_negative():
    """Test that negative terms yield a negative score."""
    score = _heuristic_sentiment_score("This is terrible and slow and broken.")
    assert score < -0.1

def test_heuristic_sentiment_score_neutral():
    """Test that unknown/empty strings yield neutral score."""
    assert _heuristic_sentiment_score("") == 0.0
    assert _heuristic_sentiment_score("The table is brown.") == 0.0

def test_heuristic_sentiment_score_case_insensitive():
    score_lower = _heuristic_sentiment_score("great")
    score_upper = _heuristic_sentiment_score("GREAT")
    assert score_lower == score_upper

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
