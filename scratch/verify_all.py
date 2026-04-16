import pandas as pd
from pathlib import Path

def check(file_path):
    if not Path(file_path).exists():
        print(f"Skipping {file_path} (not found)")
        return
    df = pd.read_csv(file_path)
    print(f"--- {file_path} ---")
    print(f"Rows: {len(df)}")
    if "sentiment_confidence" in df.columns:
        print(f"Avg Confidence: {df['sentiment_confidence'].mean():.4f}")
    if "sentiment_score" in df.columns:
        print(f"Avg Abs Score: {df['sentiment_score'].abs().mean():.4f}")

check("artifacts/sentiment/sentiment_scores.csv")
check("gnn_preprocessed/messages_nodes.csv")
