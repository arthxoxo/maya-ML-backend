"""
Train a GRU model over RoBERTa message embeddings to predict context-aware sentiment.

This script extracts RoBERTa embeddings for messages, constructs conversational
sequences, and trains a PyTorch GRU to predict the pseudo-labeled sentiment,
allowing the model to internalize technical heuristics and conversational context.
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from transformers import AutoModel, AutoTokenizer
from tqdm import tqdm

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from app_config import RAW_DATA_DIR, SECRET_DATA_DIR, SENTIMENT_ARTIFACT_DIR
from lib.device_utils import resolve_device
from lib.online_store import load_artifact_df


class ContextSentimentGRU(nn.Module):
    def __init__(self, input_dim: int = 768, hidden_dim: int = 128):
        super().__init__()
        self.gru = nn.GRU(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=1,
            batch_first=True,
            dropout=0.0
        )
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 3) # Output: Negative (0), Neutral (1), Positive (2)
        )

    def forward(self, seq):
        # seq: (batch, seq_len, input_dim)
        out, _ = self.gru(seq)
        # We only care about the sentiment of the LAST message in the sequence
        last_hidden = out[:, -1, :]
        logits = self.classifier(last_hidden)
        return logits


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--seq_len", type=int, default=3)
    p.add_argument("--max_samples", type=int, default=5000, help="Max samples to train on for speed")
    return p.parse_args()


def load_data():
    csv_path = SECRET_DATA_DIR / "whatsapp_messages.csv"
    if not csv_path.exists():
        csv_path = RAW_DATA_DIR / "whatsapp_messages.csv"
        
    df = pd.read_csv(csv_path)
    if "role" in df.columns:
        df = df[df["role"].fillna("").str.lower() == "user"].copy()
        
    # Load pseudo-labels from existing sentiment scores
    labels_df = load_artifact_df("sentiment_scores", fallback_csv_path=SENTIMENT_ARTIFACT_DIR / "sentiment_scores.csv")
    if labels_df.empty:
        raise FileNotFoundError("sentiment_scores.csv not found. Run pipeline-recompute first.")
        
    # Merge to get labels
    # Assuming order is preserved or we can merge on message
    df["message_clean"] = df["message"].fillna("").astype(str).str.strip().str.lower()
    labels_df["message_clean"] = labels_df["message"].fillna("").astype(str).str.strip().str.lower()
    
    # Merge and keep first match to avoid explosion
    df = df.merge(labels_df[["message_clean", "sentiment_label"]].drop_duplicates("message_clean"), 
                  on="message_clean", how="inner")
                  
    if "session_id" in df.columns and "created_at" in df.columns:
        df = df.sort_values(["session_id", "created_at"]).reset_index(drop=True)
        
    return df


def extract_embeddings(texts: list[str], device: str) -> np.ndarray:
    print(f"Extracting RoBERTa embeddings for {len(texts)} messages...")
    model_id = "cardiffnlp/twitter-roberta-base-sentiment-latest"
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModel.from_pretrained(model_id).to(device)
    model.eval()
    
    embeddings = []
    batch_size = 64
    with torch.no_grad():
        for i in tqdm(range(0, len(texts), batch_size)):
            batch_texts = [t[:512] for t in texts[i:i+batch_size]]
            inputs = tokenizer(batch_texts, padding=True, truncation=True, return_tensors="pt").to(device)
            outputs = model(**inputs)
            # Use [CLS] token embedding
            cls_embeddings = outputs.last_hidden_state[:, 0, :].cpu().numpy()
            embeddings.append(cls_embeddings)
            
    return np.vstack(embeddings)


def main():
    args = parse_args()
    device = resolve_device()
    
    df = load_data()
    if len(df) > args.max_samples:
        # Take a chronological slice to preserve sessions
        df = df.tail(args.max_samples).copy().reset_index(drop=True)
        
    texts = df["message"].fillna("").astype(str).tolist()
    
    # Map labels: negative=0, neutral=1, positive=2
    label_map = {"negative": 0, "neutral": 1, "positive": 2}
    y_raw = df["sentiment_label"].fillna("neutral").str.lower().map(label_map).fillna(1).astype(int).values
    
    # Extract embeddings
    embeddings = extract_embeddings(texts, device)
    
    # Build sequences (using session_id if available)
    seq_len = args.seq_len
    X_seq = []
    y_seq = []
    
    session_ids = df["session_id"].values if "session_id" in df.columns else np.zeros(len(df))
    
    for i in range(len(embeddings)):
        # Build a sequence of length seq_len ending at i
        seq = []
        for j in range(max(0, i - seq_len + 1), i + 1):
            if session_ids[j] == session_ids[i]:
                seq.append(embeddings[j])
        
        # Pad with zeros if sequence is too short
        while len(seq) < seq_len:
            seq.insert(0, np.zeros(768))
            
        X_seq.append(np.vstack(seq))
        y_seq.append(y_raw[i])
        
    X = torch.tensor(np.stack(X_seq), dtype=torch.float32)
    y = torch.tensor(y_seq, dtype=torch.long)
    
    # Train/Val split
    split_idx = int(0.8 * len(X))
    X_train, y_train = X[:split_idx], y[:split_idx]
    X_val, y_val = X[split_idx:], y[split_idx:]
    
    train_loader = DataLoader(TensorDataset(X_train, y_train), batch_size=args.batch_size, shuffle=True)
    
    model = ContextSentimentGRU(input_dim=768, hidden_dim=128).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    criterion = nn.CrossEntropyLoss()
    
    print(f"Training GRU on {len(X_train)} samples, validating on {len(X_val)} samples...")
    best_val_loss = float('inf')
    
    for epoch in range(1, args.epochs + 1):
        model.train()
        train_loss = 0.0
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            pred = model(xb)
            loss = criterion(pred, yb)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * len(xb)
        train_loss /= len(X_train)
        
        model.eval()
        with torch.no_grad():
            val_pred = model(X_val.to(device))
            val_loss = criterion(val_pred, y_val.to(device)).item()
            preds = torch.argmax(val_pred, dim=1)
            acc = (preds == y_val.to(device)).float().mean().item()
            
        print(f"Epoch {epoch:02d}/{args.epochs} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | Val Acc: {acc:.4f}")
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            out_path = SENTIMENT_ARTIFACT_DIR / "context_sentiment_gru.pt"
            out_path.parent.mkdir(parents=True, exist_ok=True)
            torch.save(model.state_dict(), out_path)
            
    print(f"Saved best model to {out_path}")

if __name__ == "__main__":
    main()
