import pandas as pd
import torch
from transformers import pipeline
from tqdm import tqdm
import time
from pathlib import Path
import sys

# Ensure project-root imports resolve
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

def main():
    csv_path = Path("secret_data/whatsapp_messages.csv")
    output_path = Path("artifacts/sentiment/sentiment_scores.csv")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if not csv_path.exists():
        print(f"Error: {csv_path} not found.")
        return

    print(f"Loading {csv_path}...")
    df = pd.read_csv(csv_path)
    # Filter for user messages only
    df = df[df["role"].fillna("").str.lower() == "user"].copy()
    print(f"Processing {len(df)} user messages...")

    # Hardware acceleration
    device = "cpu"
    if torch.backends.mps.is_available():
        device = "mps"
    elif torch.cuda.is_available():
        device = 0
    print(f"Using device: {device}")

    # Load model
    print("Initializing CardiffNLP RoBERTa model...")
    pipe = pipeline(
        "sentiment-analysis",
        model="cardiffnlp/twitter-roberta-base-sentiment-latest",
        tokenizer="cardiffnlp/twitter-roberta-base-sentiment-latest",
        device=device
    )

    batch_size = 32
    texts = df["message"].fillna("").astype(str).tolist()
    results = []

    start_time = time.time()
    for i in tqdm(range(0, len(texts), batch_size)):
        batch = texts[i : i + batch_size]
        # RoBERTa limit is 512 tokens
        batch = [t[:512] for t in batch]
        out = pipe(batch, truncation=True, max_length=256)
        results.extend(out)
    
    end_time = time.time()
    print(f"\nInference complete in {end_time - start_time:.2f}s")

    # Map results to polarized scores and confidence
    scores = []
    confs = []
    labels = []

    for r in results:
        label = r["label"].lower()
        score = r["score"] # This is the softmax probability
        
        confs.append(round(score, 4))
        labels.append(label)
        
        if "positive" in label or label == "label_2":
            scores.append(round(score, 4))
        elif "negative" in label or label == "label_0":
            scores.append(round(-score, 4))
        else:
            scores.append(0.0)

    df["sentiment_score"] = scores
    df["sentiment_confidence"] = confs
    df["sentiment_label"] = labels

    print(f"Saving results to {output_path}...")
    df.to_csv(output_path, index=False)
    print("Bulk processing ready for dashboard visualization.")

if __name__ == "__main__":
    main()
