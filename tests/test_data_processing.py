import sys
from pathlib import Path
import pandas as pd
import pytest
from unittest.mock import patch

# Ensure project-root imports resolve
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from pipelines.preprocessing.build_gnn_nodes_from_flink import normalize_raw_table, align_columns

def test_align_columns():
    """Test that align_columns correctly pads/reorders columns."""
    df = pd.DataFrame({"a": [1, 2], "b": [3, 4]})
    expected = ["b", "c", "a"]
    aligned = align_columns(df, expected)
    
    assert list(aligned.columns) == expected
    assert pd.isna(aligned.loc[0, "c"])
    assert aligned.loc[0, "b"] == 3

def test_normalize_raw_table_headerless():
    """Test normalization of headerless Flink output (numbered columns)."""
    # Simulate a 2-column headerless CSV read
    df = pd.DataFrame({0: [101, 102], 1: ["msg1", "msg2"]})
    expected_cols = ["id", "content", "extra"]
    
    norm = normalize_raw_table(df, expected_cols)
    assert list(norm.columns) == expected_cols
    assert norm.loc[0, "id"] == 101
    assert norm.loc[0, "content"] == "msg1"
    assert pd.isna(norm.loc[0, "extra"])

@patch("apps.dashboard.streamlit_dashboard.pd.read_csv")
@patch("apps.dashboard.streamlit_dashboard.SESSIONS_SOURCE_PATH")
@patch("apps.dashboard.streamlit_dashboard.PREPROCESSED_DIR")
def test_load_whatsapp_sentiment_messages_join(mock_preproc_dir, mock_sessions_path, mock_read_csv):
    """Test the session joining logic in load_whatsapp_sentiment_messages."""
    from apps.dashboard.streamlit_dashboard import load_whatsapp_sentiment_messages
    
    # Setup mocks
    mock_preproc_dir.__truediv__.return_value.exists.return_value = True
    mock_sessions_path.exists.return_value = True
    
    # First call to read_csv (messages_nodes)
    msg_df = pd.DataFrame({
        "message_id": [1, 2],
        "session_id": [10, 20],
        "message": ["hi", "bye"],
        "created_at": ["2026-01-01", "2026-01-02"],
        "sentiment_score": [0.0, 0.0],
        "sentiment_label": ["neutral", "neutral"]
    })
    # Second call to read_csv (sessions.csv fallback)
    sess_df = pd.DataFrame({
        "id": [10, 20],
        "user_id": [999, 888]
    })
    
    mock_read_csv.side_effect = [msg_df, sess_df]
    
    # We also need to patch SENTIMENT_SCORES_PATH to trigger fallback
    with patch("apps.dashboard.streamlit_dashboard.SENTIMENT_SCORES_PATH") as mock_sent_path:
        mock_sent_path.exists.return_value = False
        res = load_whatsapp_sentiment_messages()
        
        # Verify user_id was joined from mock sessions
        assert res.loc[0, "user_id"] == 999
        assert res.loc[1, "user_id"] == 888
        assert res.loc[0, "message"] == "hi"

def test_normalize_raw_table_headered():
    """Test normalization of headered CSV (keep known, add missing)."""
    df = pd.DataFrame({"id": [1], "unknown": ["x"]})
    expected = ["id", "extra"]
    norm = normalize_raw_table(df, expected)
    
    assert "id" in norm.columns
    assert "extra" in norm.columns
    assert "unknown" not in norm.columns
