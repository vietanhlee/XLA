"""
Utility functions for metrics, checkpointing, and visualization.
Paper: "Zero-Shot Detection of AI-Generated Images" (Cozzolino et al.)
"""

import os
from pathlib import Path
from typing import Tuple, Dict, Any, List, Optional

import torch
import numpy as np
from sklearn.metrics import roc_auc_score, roc_curve

def compute_metrics(scores: np.ndarray, labels: np.ndarray) -> Dict[str, float]:
    """
    Computes ROC-AUC score, Best Balanced Accuracy, and optimal Decision Threshold.
    
    Args:
        scores: Array of decision statistics (e.g. D^(0) or |Delta^01|) for test samples.
                Higher score indicates higher anomaly / likelihood of being AI-generated.
        labels: Binary labels array (0 = Real, 1 = Fake/AI-generated).
        
    Returns:
        Dict containing:
            - 'auc': Area Under the ROC Curve (%)
            - 'best_acc': Best balanced accuracy (%) across all threshold settings
            - 'best_threshold': Optimal decision threshold value
    """
    if len(np.unique(labels)) < 2:
        return {"auc": 0.0, "best_acc": 0.0, "best_threshold": 0.0}

    auc = roc_auc_score(labels, scores) * 100.0
    
    # Calculate ROC curve to find optimal threshold maximizing Balanced Accuracy
    fpr, tpr, thresholds = roc_curve(labels, scores)
    balanced_accs = (tpr + (1.0 - fpr)) / 2.0
    best_idx = np.argmax(balanced_accs)
    
    best_acc = balanced_accs[best_idx] * 100.0
    best_threshold = thresholds[best_idx]

    return {
        "auc": float(auc),
        "best_acc": float(best_acc),
        "best_threshold": float(best_threshold)
    }

def save_checkpoint(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    loss: float,
    filepath: str,
    scheduler: Optional[Any] = None
):
    """Saves model state and training state to file."""
    Path(filepath).parent.mkdir(parents=True, exist_ok=True)
    checkpoint = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "loss": loss
    }
    if scheduler is not None:
        checkpoint["scheduler_state_dict"] = scheduler.state_dict()
    torch.save(checkpoint, filepath)
    print(f"Checkpoint saved successfully to: {filepath}")

def load_checkpoint(
    filepath: str,
    model: torch.nn.Module,
    optimizer: Optional[torch.optim.Optimizer] = None,
    scheduler: Optional[Any] = None,
    device: str = "cpu"
) -> Dict[str, Any]:
    """Loads checkpoint into model, optimizer, and scheduler."""
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Checkpoint not found at path: {filepath}")
        
    checkpoint = torch.load(filepath, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    if optimizer is not None and "optimizer_state_dict" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    if scheduler is not None and "scheduler_state_dict" in checkpoint:
        scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        
    print(f"Loaded checkpoint from: {filepath} (Epoch {checkpoint.get('epoch', 0)})")
    return checkpoint


class EarlyStopping:
    """
    Early stopping helper to stop training when validation loss does not improve 
    after a specified number of patience epochs.
    """

    def __init__(self, patience: int = 7, min_delta: float = 1e-4, verbose: bool = True):
        self.patience = patience
        self.min_delta = min_delta
        self.verbose = verbose
        self.counter = 0
        self.best_loss = float("inf")
        self.early_stop = False

    def __call__(self, val_loss: float) -> bool:
        """
        Returns True if new val_loss is an improvement (lower by at least min_delta), 
        or False otherwise.
        """
        if val_loss < (self.best_loss - self.min_delta):
            if self.verbose and self.best_loss != float("inf"):
                print(f"★ Validation loss improved from {self.best_loss:.4f} to {val_loss:.4f}.")
            self.best_loss = val_loss
            self.counter = 0
            return True
        else:
            self.counter += 1
            if self.verbose:
                print(f"⚠ EarlyStopping counter: {self.counter}/{self.patience} (Best val loss: {self.best_loss:.4f})")
            if self.counter >= self.patience:
                self.early_stop = True
            return False

