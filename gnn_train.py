"""
GNN Training Pipeline — Train HeteroGraphSAGE + Feature Importance

Pipeline:
  1. Load graph_data.pt (heterogeneous graph)
  2. Create train/val masks on User nodes
  3. Train HeteroGraphSAGE model
  4. Compute per-user feature importance via GNNExplainer
  5. Save model checkpoint + feature importance CSV

Usage:
    source flink_venv/bin/activate
    python gnn_train.py
"""

import torch
import torch.nn.functional as F
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import warnings
import time
import json

warnings.filterwarnings("ignore")

from gnn_model import HeteroGraphSAGE

# ── Configuration ────────────────────────────────────────────────────────────

OUTPUT_DIR = Path("/Users/arthxoxo/maya-ML-backend")
GRAPH_PATH = OUTPUT_DIR / "graph_data.pt"
MODEL_PATH = OUTPUT_DIR / "gnn_model.pt"
IMPORTANCE_PATH = OUTPUT_DIR / "gnn_feature_importance.csv"
EMBEDDINGS_PATH = OUTPUT_DIR / "gnn_user_embeddings.csv"
METRICS_PATH = OUTPUT_DIR / "gnn_training_metrics.json"

# Hyperparameters
HIDDEN_DIM = 64
OUT_DIM = 32
DROPOUT = 0.3
LEARNING_RATE = 0.01
WEIGHT_DECAY = 5e-4
EPOCHS = 300
PATIENCE = 30
TRAIN_RATIO = 0.8

# ── Data Loading ─────────────────────────────────────────────────────────────


def load_graph():
    """Load the heterogeneous graph data."""
    print("📂  Loading graph data...")
    data = torch.load(GRAPH_PATH, weights_only=False)

    print(f"    Node types: {data.node_types}")
    print(f"    Edge types: {len(data.edge_types)}")
    print(f"    User nodes: {data['user'].x.shape[0]}")
    print(f"    Target: engagement_score")

    return data


# ── Train/Val Split ──────────────────────────────────────────────────────────


def create_masks(num_users, train_ratio=TRAIN_RATIO):
    """Create random train/val masks for user nodes."""
    perm = torch.randperm(num_users)
    train_size = int(num_users * train_ratio)

    train_mask = torch.zeros(num_users, dtype=torch.bool)
    val_mask = torch.zeros(num_users, dtype=torch.bool)

    train_mask[perm[:train_size]] = True
    val_mask[perm[train_size:]] = True

    print(f"    Train: {train_mask.sum().item()} users")
    print(f"    Val:   {val_mask.sum().item()} users")

    return train_mask, val_mask


# ── Training Loop ────────────────────────────────────────────────────────────


def train_epoch(model, optimizer, data, train_mask):
    """Run one training epoch."""
    model.train()
    optimizer.zero_grad()

    x_dict = {nt: data[nt].x for nt in data.node_types}
    edge_index_dict = {et: data[et].edge_index for et in data.edge_types}

    predictions, _ = model(x_dict, edge_index_dict)

    # Handle NaN predictions
    preds = predictions[train_mask]
    targets = data["user"].y[train_mask]
    valid = ~torch.isnan(preds) & ~torch.isnan(targets)
    if valid.sum() == 0:
        return float('nan')

    loss = F.mse_loss(preds[valid], targets[valid])

    if torch.isnan(loss):
        return float('nan')

    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
    optimizer.step()

    return loss.item()


@torch.no_grad()
def evaluate(model, data, mask):
    """Evaluate the model on a subset of users."""
    model.eval()

    x_dict = {nt: data[nt].x for nt in data.node_types}
    edge_index_dict = {et: data[et].edge_index for et in data.edge_types}

    predictions, _ = model(x_dict, edge_index_dict)

    preds = predictions[mask].numpy()
    labels = data["user"].y[mask].numpy()

    # Replace NaN with mean prediction to avoid sklearn errors
    preds = np.nan_to_num(preds, nan=float(np.nanmean(preds)) if not np.all(np.isnan(preds)) else 0.0)
    labels = np.nan_to_num(labels, nan=0.0)

    mse = mean_squared_error(labels, preds)
    mae = mean_absolute_error(labels, preds)
    r2 = r2_score(labels, preds) if len(labels) > 1 else 0.0

    return mse, mae, r2, preds, labels


# ── Feature Importance via Perturbation ──────────────────────────────────────


def compute_feature_importance(model, data):
    """
    Compute per-user feature importance via input perturbation.

    For each user feature, we perturb it to zero and measure the change
    in prediction — this gives a simple, interpretable importance score.
    """
    print("\n🔍  Computing GNN feature importance (perturbation-based)...")

    model.eval()
    feature_cols = data.user_feature_cols
    num_users = data["user"].x.shape[0]
    num_features = len(feature_cols)

    x_dict = {nt: data[nt].x for nt in data.node_types}
    edge_index_dict = {et: data[et].edge_index for et in data.edge_types}

    # Baseline predictions
    with torch.no_grad():
        baseline_preds, _ = model(x_dict, edge_index_dict)
        baseline_preds = baseline_preds.numpy()

    # Per-feature perturbation importance
    importance_matrix = np.zeros((num_users, num_features))

    for feat_idx in range(num_features):
        # Create perturbed input (zero out this feature for all users)
        perturbed_x = data["user"].x.clone()
        perturbed_x[:, feat_idx] = 0.0

        perturbed_x_dict = dict(x_dict)
        perturbed_x_dict["user"] = perturbed_x

        with torch.no_grad():
            perturbed_preds, _ = model(perturbed_x_dict, edge_index_dict)
            perturbed_preds = perturbed_preds.numpy()

        # Importance = absolute change in prediction
        importance_matrix[:, feat_idx] = np.abs(baseline_preds - perturbed_preds)

    # Build DataFrame
    user_names = data.user_names
    user_ids_ordered = [uid for uid, _ in sorted(
        data.user_id_to_idx.items(), key=lambda x: x[1]
    )]

    rows = []
    for user_idx in range(num_users):
        user_importance = importance_matrix[user_idx]
        # Normalize per user
        total = user_importance.sum()
        if total > 0:
            user_importance_norm = user_importance / total
        else:
            user_importance_norm = user_importance

        for feat_idx, feat_name in enumerate(feature_cols):
            rows.append({
                "user_id": user_ids_ordered[user_idx],
                "user_name": user_names[user_idx],
                "feature": feat_name,
                "importance_raw": float(importance_matrix[user_idx, feat_idx]),
                "importance_normalized": float(user_importance_norm[feat_idx]),
                "predicted_engagement": float(baseline_preds[user_idx]),
            })

    importance_df = pd.DataFrame(rows)

    # Global importance (mean across users)
    global_importance = importance_df.groupby("feature")["importance_raw"].mean()
    global_importance = global_importance.sort_values(ascending=False)

    print(f"\n📊  Top 10 Most Important Features (GNN):")
    for feat, imp in global_importance.head(10).items():
        print(f"    {feat:35s}  {imp:.6f}")

    return importance_df, global_importance


# ── Save User Embeddings ─────────────────────────────────────────────────────


def save_user_embeddings(model, data):
    """Extract and save user embeddings for visualization."""
    print("\n💾  Extracting user embeddings...")

    model.eval()
    x_dict = {nt: data[nt].x for nt in data.node_types}
    edge_index_dict = {et: data[et].edge_index for et in data.edge_types}

    with torch.no_grad():
        _, h_dict = model(x_dict, edge_index_dict)
        user_embeds = h_dict["user"].numpy()

    user_ids_ordered = [uid for uid, _ in sorted(
        data.user_id_to_idx.items(), key=lambda x: x[1]
    )]

    embed_df = pd.DataFrame(
        user_embeds,
        columns=[f"gnn_embed_{i}" for i in range(user_embeds.shape[1])],
    )
    embed_df.insert(0, "user_id", user_ids_ordered)
    embed_df.insert(1, "user_name", data.user_names)

    embed_df.to_csv(EMBEDDINGS_PATH, index=False)
    print(f"    ✅ Saved {len(embed_df)} user embeddings to {EMBEDDINGS_PATH}")

    return embed_df


# ── Main Training ────────────────────────────────────────────────────────────


def main():
    print("=" * 60)
    print("  Maya ML — GNN Training Pipeline")
    print("=" * 60)

    # ── Load ─────────────────────────────────────────────────────────────
    data = load_graph()

    # ── Masks ────────────────────────────────────────────────────────────
    num_users = data["user"].x.shape[0]
    train_mask, val_mask = create_masks(num_users)

    # ── Model ────────────────────────────────────────────────────────────
    in_channels_dict = {nt: data[nt].x.shape[1] for nt in data.node_types}

    model = HeteroGraphSAGE(
        metadata=(data.node_types, data.edge_types),
        in_channels_dict=in_channels_dict,
        hidden_channels=HIDDEN_DIM,
        out_channels=OUT_DIM,
        dropout=DROPOUT,
    )

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\n🧠  Model: HeteroGraphSAGE")
    print(f"    Total params:     {total_params:>8,}")
    print(f"    Trainable params: {trainable_params:>8,}")

    optimizer = torch.optim.Adam(
        model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY
    )

    # ── Training loop ────────────────────────────────────────────────────
    print(f"\n🚀  Training for up to {EPOCHS} epochs (patience={PATIENCE})...")
    print(f"    lr={LEARNING_RATE}, weight_decay={WEIGHT_DECAY}")
    print(f"{'─' * 60}")

    best_val_mse = float("inf")
    best_epoch = 0
    patience_counter = 0
    history = {"train_loss": [], "val_mse": [], "val_mae": [], "val_r2": []}

    start_time = time.time()

    for epoch in range(1, EPOCHS + 1):
        # Train
        train_loss = train_epoch(model, optimizer, data, train_mask)

        # Skip if NaN loss
        if np.isnan(train_loss):
            print(f"    Epoch {epoch:>3d} │ NaN loss detected, skipping...")
            continue

        # Evaluate
        val_mse, val_mae, val_r2, _, _ = evaluate(model, data, val_mask)
        train_mse, train_mae, train_r2, _, _ = evaluate(model, data, train_mask)

        history["train_loss"].append(train_loss)
        history["val_mse"].append(val_mse)
        history["val_mae"].append(val_mae)
        history["val_r2"].append(val_r2)

        # Early stopping
        if val_mse < best_val_mse:
            best_val_mse = val_mse
            best_epoch = epoch
            patience_counter = 0
            # Save best model
            torch.save({
                "model_state_dict": model.state_dict(),
                "in_channels_dict": in_channels_dict,
                "metadata": (data.node_types, data.edge_types),
                "hidden_channels": HIDDEN_DIM,
                "out_channels": OUT_DIM,
                "dropout": DROPOUT,
                "epoch": epoch,
                "val_mse": val_mse,
            }, MODEL_PATH)
        else:
            patience_counter += 1

        if epoch % 10 == 0 or epoch == 1 or patience_counter == PATIENCE:
            print(
                f"    Epoch {epoch:>3d} │ "
                f"Loss: {train_loss:.4f} │ "
                f"Train MSE: {train_mse:.4f} │ "
                f"Val MSE: {val_mse:.4f} │ "
                f"Val R²: {val_r2:.4f} │ "
                f"{'★' if epoch == best_epoch else ' '}"
            )

        if patience_counter >= PATIENCE:
            print(f"\n    ⏹  Early stopping at epoch {epoch} (best: {best_epoch})")
            break

    elapsed = time.time() - start_time

    # ── Load best model ──────────────────────────────────────────────────
    checkpoint = torch.load(MODEL_PATH, weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    print(f"\n    ✅ Loaded best model from epoch {checkpoint['epoch']}")

    # ── Final evaluation ─────────────────────────────────────────────────
    print(f"\n{'═' * 60}")
    print(f"📊  Final Evaluation:")

    train_mse, train_mae, train_r2, train_preds, train_labels = evaluate(
        model, data, train_mask
    )
    val_mse, val_mae, val_r2, val_preds, val_labels = evaluate(
        model, data, val_mask
    )

    # Full dataset evaluation
    all_mask = torch.ones(num_users, dtype=torch.bool)
    all_mse, all_mae, all_r2, all_preds, all_labels = evaluate(
        model, data, all_mask
    )

    print(f"    {'':15s} {'MSE':>8s} {'MAE':>8s} {'R²':>8s}")
    print(f"    {'─' * 45}")
    print(f"    {'Train':15s} {train_mse:>8.4f} {train_mae:>8.4f} {train_r2:>8.4f}")
    print(f"    {'Validation':15s} {val_mse:>8.4f} {val_mae:>8.4f} {val_r2:>8.4f}")
    print(f"    {'All Users':15s} {all_mse:>8.4f} {all_mae:>8.4f} {all_r2:>8.4f}")
    print(f"\n    Training time: {elapsed:.1f}s")
    print(f"    Best epoch: {best_epoch}")
    print(f"    Model saved: {MODEL_PATH}")

    # Predictions vs actual
    print(f"\n📋  Sample Predictions (All Users):")
    print(f"    {'User':25s} {'Actual':>8s} {'Predicted':>10s} {'Error':>8s}")
    print(f"    {'─' * 55}")
    user_names = data.user_names
    user_ids_ordered = [uid for uid, _ in sorted(
        data.user_id_to_idx.items(), key=lambda x: x[1]
    )]

    # Show top 15 users by actual engagement
    indices = np.argsort(all_labels)[::-1][:15]
    for idx in indices:
        name = user_names[idx][:24]
        actual = all_labels[idx]
        pred = all_preds[idx]
        error = abs(actual - pred)
        print(f"    {name:25s} {actual:>8.4f} {pred:>10.4f} {error:>8.4f}")

    # ── Feature importance ───────────────────────────────────────────────
    importance_df, global_importance = compute_feature_importance(model, data)
    importance_df.to_csv(IMPORTANCE_PATH, index=False)
    print(f"\n💾  Feature importance saved: {IMPORTANCE_PATH}")

    # ── User embeddings ──────────────────────────────────────────────────
    save_user_embeddings(model, data)

    # ── Save training metrics ────────────────────────────────────────────
    metrics = {
        "train_mse": float(train_mse),
        "train_mae": float(train_mae),
        "train_r2": float(train_r2),
        "val_mse": float(val_mse),
        "val_mae": float(val_mae),
        "val_r2": float(val_r2),
        "all_mse": float(all_mse),
        "all_mae": float(all_mae),
        "all_r2": float(all_r2),
        "best_epoch": best_epoch,
        "total_epochs": epoch,
        "training_time_sec": round(elapsed, 1),
        "num_users": num_users,
        "num_features": len(data.user_feature_cols),
        "history": history,
    }
    with open(METRICS_PATH, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"💾  Training metrics saved: {METRICS_PATH}")

    print(f"\n{'═' * 60}")
    print(f"✅  GNN Training Pipeline Complete!")
    print(f"{'═' * 60}")


if __name__ == "__main__":
    main()
