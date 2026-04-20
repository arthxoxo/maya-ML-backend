"""
device_utils.py — Shared hardware-acceleration helpers for the Maya ML pipeline.

Usage:
    from lib.device_utils import resolve_device, resolve_xgb_device

    # PyTorch models / HuggingFace pipelines
    device = resolve_device()          # torch.device("mps") | "cuda" | "cpu"
    model.to(device)

    # XGBoost
    xgb_device = resolve_xgb_device() # "mps" | "cuda" | "cpu"
    XGBClassifier(device=xgb_device, ...)
"""

from __future__ import annotations

import torch


def resolve_device() -> torch.device:
    """Return the best available torch device: CUDA > MPS > CPU.

    Priority:
        1. CUDA  — Nvidia GPU (Linux / Windows workstation)
        2. MPS   — Apple Silicon GPU (macOS M-series)
        3. CPU   — Universal fallback
    """
    if torch.cuda.is_available():
        dev = torch.device("cuda")
    elif torch.backends.mps.is_available():
        dev = torch.device("mps")
    else:
        dev = torch.device("cpu")

    print(f"[device] Using torch device: {dev}")
    return dev


def resolve_xgb_device() -> str:
    """Return the XGBoost-compatible device string: 'cuda' | 'cpu'.

    Note: While some XGBoost builds support 'mps', the current environment 
    build requires falling back to 'cpu' for stability on macOS.
    """
    if torch.cuda.is_available():
        xgb_dev = "cuda"
    else:
        # Fallback to cpu for Mac/MPS since current XGB build lacks 'mps' support
        xgb_dev = "cpu"

    print(f"[device] Using XGBoost device: {xgb_dev}")
    return xgb_dev
