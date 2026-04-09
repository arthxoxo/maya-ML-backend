"""
GNN Model — HeteroGraphSAGE for Maya Behavioral Prediction

Architecture:
  Input (per-node type features)
    → HeteroConv Layer 1 (SAGEConv per edge type, hidden_dim=64)
    → LeakyReLU + Dropout(0.3)
    → HeteroConv Layer 2 (SAGEConv, hidden_dim=32)
    → LeakyReLU
    → Linear Head (32 → 1) on User nodes → engagement prediction

Supports:
  - Forward pass for training/inference
  - Get user embeddings for downstream analysis
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import SAGEConv, HeteroConv, Linear


class HeteroGraphSAGE(nn.Module):
    """
    Heterogeneous GraphSAGE for User engagement prediction.

    Uses HeteroConv to apply SAGEConv independently per edge type,
    then aggregates neighbor messages for each node type.
    """

    def __init__(self, metadata, in_channels_dict, hidden_channels=64,
                 out_channels=32, dropout=0.3):
        """
        Args:
            metadata: (node_types, edge_types) from HeteroData
            in_channels_dict: dict mapping node_type → number of input features
            hidden_channels: hidden layer size (default: 64)
            out_channels: output embedding size (default: 32)
            dropout: dropout rate (default: 0.3)
        """
        super().__init__()

        self.dropout = dropout
        node_types, edge_types = metadata

        # ── Input projections (align all node types to hidden_channels) ──
        self.input_projections = nn.ModuleDict()
        for node_type in node_types:
            in_ch = in_channels_dict[node_type]
            self.input_projections[node_type] = Linear(in_ch, hidden_channels)

        # ── HeteroConv Layer 1 ───────────────────────────────────────────
        conv1_dict = {}
        for edge_type in edge_types:
            conv1_dict[edge_type] = SAGEConv(
                (hidden_channels, hidden_channels),
                hidden_channels,
            )
        self.conv1 = HeteroConv(conv1_dict, aggr="mean")

        # ── HeteroConv Layer 2 ───────────────────────────────────────────
        conv2_dict = {}
        for edge_type in edge_types:
            conv2_dict[edge_type] = SAGEConv(
                (hidden_channels, hidden_channels),
                out_channels,
            )
        self.conv2 = HeteroConv(conv2_dict, aggr="mean")

        # ── Prediction head (User nodes only) ────────────────────────────
        self.pred_head = nn.Sequential(
            nn.Linear(out_channels, 16),
            nn.LeakyReLU(0.1),
            nn.Dropout(dropout),
            nn.Linear(16, 1),
            nn.Sigmoid(),  # engagement_score is in [0, 1]
        )

    def _safe_clamp(self, h_dict):
        """Clamp values and replace NaN to prevent gradient explosions."""
        out = {}
        for key, h in h_dict.items():
            h = torch.nan_to_num(h, nan=0.0, posinf=1.0, neginf=-1.0)
            h = h.clamp(-10, 10)
            out[key] = h
        return out

    def forward(self, x_dict, edge_index_dict):
        """
        Forward pass through the heterogeneous GNN.

        Args:
            x_dict: dict node_type → feature tensor
            edge_index_dict: dict edge_type → edge_index tensor

        Returns:
            predictions: tensor of shape (num_users,) — engagement predictions
            embeddings: dict node_type → embedding tensor
        """
        # ── Project inputs ───────────────────────────────────────────────
        h_dict = {}
        for node_type, x in x_dict.items():
            x = torch.nan_to_num(x, nan=0.0)
            h_dict[node_type] = self.input_projections[node_type](x)
        h_dict = self._safe_clamp(h_dict)

        # ── Layer 1 ──────────────────────────────────────────────────────
        h_dict = self.conv1(h_dict, edge_index_dict)
        h_dict = self._safe_clamp(h_dict)
        h_dict = {
            key: F.leaky_relu(h, 0.1)
            for key, h in h_dict.items()
        }
        h_dict = {
            key: F.dropout(h, p=self.dropout, training=self.training)
            for key, h in h_dict.items()
        }

        # ── Layer 2 ──────────────────────────────────────────────────────
        h_dict = self.conv2(h_dict, edge_index_dict)
        h_dict = self._safe_clamp(h_dict)
        h_dict = {
            key: F.leaky_relu(h, 0.1)
            for key, h in h_dict.items()
        }

        # ── Prediction (User nodes only) ─────────────────────────────────
        user_embeddings = h_dict.get("user")
        if user_embeddings is not None:
            predictions = self.pred_head(user_embeddings).squeeze(-1)
        else:
            predictions = None

        return predictions, h_dict

    def get_user_embeddings(self, x_dict, edge_index_dict):
        """Get user node embeddings for downstream analysis."""
        self.eval()
        with torch.no_grad():
            _, h_dict = self.forward(x_dict, edge_index_dict)
        return h_dict.get("user")
