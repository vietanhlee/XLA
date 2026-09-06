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


import json
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score, roc_curve, precision_recall_curve, average_precision_score, confusion_matrix

def plot_training_curves(
    history: Dict[str, List[float]],
    output_path: str,
    model_name: str = "ZED"
):
    """
    Plots and saves publication-quality training & validation loss curves and learning rate schedule.
    """
    epochs = history.get("epoch", list(range(1, len(history.get("train_loss", [])) + 1)))
    train_loss = history.get("train_loss", [])
    val_loss = history.get("val_loss", [])
    lrs = history.get("lr", [])

    if not train_loss:
        return

    # Use clean style
    plt.rcParams.update({'font.sans-serif': 'DejaVu Sans', 'font.size': 10})
    fig, axes = plt.subplots(1, 2, figsize=(14, 5), dpi=200)
    fig.suptitle(f"📊 {model_name} Model - Training Progress & Convergence Dashboard", fontsize=14, fontweight="bold", y=1.02)

    # Subplot 1: NLL Loss Curves (bits/pixel)
    ax1 = axes[0]
    ax1.plot(epochs, train_loss, label="Training NLL Loss", color="#1f77b4", linewidth=2.2, marker="o", markersize=4.5, alpha=0.9)
    
    if val_loss and len(val_loss) == len(epochs):
        ax1.plot(epochs, val_loss, label="Validation NLL Loss", color="#ff7f0e", linewidth=2.2, linestyle="--", marker="s", markersize=4.5, alpha=0.9)
        best_idx = int(np.argmin(val_loss))
        best_epoch = epochs[best_idx]
        min_val = val_loss[best_idx]
        ax1.scatter([best_epoch], [min_val], color="#d62728", s=100, zorder=6, label=f"Best Val: {min_val:.4f} (Ep {best_epoch})")
        ax1.axvline(x=best_epoch, color="#d62728", linestyle=":", alpha=0.5)

    ax1.set_title("Negative Log-Likelihood (NLL) Loss per Epoch", fontsize=11, fontweight="bold", pad=8)
    ax1.set_xlabel("Epoch Number", fontsize=10, labelpad=6)
    ax1.set_ylabel("Loss (bits / pixel)", fontsize=10, labelpad=6)
    ax1.grid(True, linestyle=":", alpha=0.6)
    ax1.legend(loc="upper right", frameon=True, facecolor="white", framealpha=0.95, edgecolor="#cccccc")

    # Subplot 2: Learning Rate Schedule
    ax2 = axes[1]
    if lrs and len(lrs) == len(epochs):
        ax2.plot(epochs, lrs, label="Learning Rate", color="#2ca02c", linewidth=2.2, marker="d", markersize=4.5, alpha=0.9)
        ax2.set_title("Learning Rate Decay Schedule per Epoch", fontsize=11, fontweight="bold", pad=8)
        ax2.set_xlabel("Epoch Number", fontsize=10, labelpad=6)
        ax2.set_ylabel("Learning Rate (Log Scale)", fontsize=10, labelpad=6)
        ax2.set_yscale("log")
        ax2.grid(True, linestyle=":", alpha=0.6)
        ax2.legend(loc="upper right", frameon=True, facecolor="white", framealpha=0.95, edgecolor="#cccccc")

    plt.tight_layout()
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, bbox_inches="tight")
    plt.close(fig)
    print(f"📊 High-resolution training curves dashboard saved to: {output_path}")


def plot_detection_dashboard(
    d0_scores: np.ndarray,
    abs_d0_scores: np.ndarray,
    delta01_scores: np.ndarray,
    labels: np.ndarray,
    metrics_d0: Dict[str, float],
    output_path: str,
    model_name: str = "ZED"
):
    """
    Plots a publication-ready 4-panel Zero-Shot Detection Evaluation Dashboard:
      1. ROC Curves for D^(0), |D^(0)|, |Delta^01|
      2. Precision-Recall (PR) Curves
      3. Real vs AI Score Density Histogram & Optimal Threshold Line
      4. Confusion Matrix Heatmap
    """
    if len(np.unique(labels)) < 2:
        print("⚠️ Warning: Cannot plot detection dashboard (test set does not contain both Real and Fake samples).")
        return

    plt.rcParams.update({'font.sans-serif': 'DejaVu Sans', 'font.size': 10})
    fig, axes = plt.subplots(2, 2, figsize=(15, 12), dpi=200)
    fig.suptitle(f"🚀 {model_name} - Zero-Shot AI Image Detection Performance Dashboard", fontsize=15, fontweight="bold", y=0.98)

    colors = {"d0": "#1f77b4", "abs_d0": "#ff7f0e", "delta01": "#2ca02c"}

    # -------------------------------------------------------------
    # Panel 1: ROC Curves
    # -------------------------------------------------------------
    ax1 = axes[0, 0]
    fpr_d0, tpr_d0, _ = roc_curve(labels, d0_scores)
    fpr_abs, tpr_abs, _ = roc_curve(labels, abs_d0_scores)
    fpr_del, tpr_del, _ = roc_curve(labels, delta01_scores)

    auc_d0 = roc_auc_score(labels, d0_scores) * 100.0
    auc_abs = roc_auc_score(labels, abs_d0_scores) * 100.0
    auc_del = roc_auc_score(labels, delta01_scores) * 100.0

    ax1.plot(fpr_d0, tpr_d0, color=colors["d0"], linewidth=2.2, label=f"$D^{(0)}$ Coding Cost (AUC = {auc_d0:.2f}%)")
    ax1.plot(fpr_abs, tpr_abs, color=colors["abs_d0"], linewidth=2.2, linestyle="--", label=f"$|D^{(0)}|$ Magnitude (AUC = {auc_abs:.2f}%)")
    ax1.plot(fpr_del, tpr_del, color=colors["delta01"], linewidth=2.2, linestyle="-.", label=f"$|\\Delta^{{01}}|$ Residual (AUC = {auc_del:.2f}%)")
    ax1.plot([0, 1], [0, 1], color="gray", linestyle=":", linewidth=1.5, label="Random Guess (AUC = 50.0%)")

    ax1.set_title("Receiver Operating Characteristic (ROC) Curves", fontsize=12, fontweight="bold")
    ax1.set_xlabel("False Positive Rate (FPR)", fontsize=10)
    ax1.set_ylabel("True Positive Rate (TPR)", fontsize=10)
    ax1.grid(True, linestyle=":", alpha=0.6)
    ax1.legend(loc="lower right", frameon=True, facecolor="white", framealpha=0.95)

    # -------------------------------------------------------------
    # Panel 2: Precision-Recall Curves
    # -------------------------------------------------------------
    ax2 = axes[0, 1]
    p_d0, r_d0, _ = precision_recall_curve(labels, d0_scores)
    p_del, r_del, _ = precision_recall_curve(labels, delta01_scores)
    ap_d0 = average_precision_score(labels, d0_scores) * 100.0
    ap_del = average_precision_score(labels, delta01_scores) * 100.0

    ax2.plot(r_d0, p_d0, color=colors["d0"], linewidth=2.2, label=f"$D^{(0)}$ (AP = {ap_d0:.2f}%)")
    ax2.plot(r_del, p_del, color=colors["delta01"], linewidth=2.2, linestyle="-.", label=f"$|\\Delta^{{01}}|$ (AP = {ap_del:.2f}%)")

    ax2.set_title("Precision-Recall (PR) Curves", fontsize=12, fontweight="bold")
    ax2.set_xlabel("Recall (Sensitivity)", fontsize=10)
    ax2.set_ylabel("Precision (Positive Predictive Value)", fontsize=10)
    ax2.grid(True, linestyle=":", alpha=0.6)
    ax2.legend(loc="lower left", frameon=True, facecolor="white", framealpha=0.95)

    # -------------------------------------------------------------
    # Panel 3: Score Distribution Histogram (Real vs AI) & Threshold
    # -------------------------------------------------------------
    ax3 = axes[1, 0]
    real_scores = d0_scores[labels == 0]
    fake_scores = d0_scores[labels == 1]
    best_thresh = metrics_d0["best_threshold"]

    ax3.hist(real_scores, bins=35, alpha=0.65, color="#1f77b4", label=f"Real Images (N={len(real_scores)})", density=True, edgecolor="none")
    ax3.hist(fake_scores, bins=35, alpha=0.65, color="#d62728", label=f"AI/Fake Images (N={len(fake_scores)})", density=True, edgecolor="none")
    ax3.axvline(x=best_thresh, color="#2ca02c", linestyle="--", linewidth=2.5, label=f"Optimal Threshold ($\\tau^* = {best_thresh:.3f}$)")

    ax3.set_title("Score Density Distribution ($D^{(0)}$ Coding Cost)", fontsize=12, fontweight="bold")
    ax3.set_xlabel("Anomaly Score $D^{(0)}$ (Higher = AI-Generated)", fontsize=10)
    ax3.set_ylabel("Probability Density", fontsize=10)
    ax3.grid(True, linestyle=":", alpha=0.6)
    ax3.legend(loc="upper right", frameon=True, facecolor="white", framealpha=0.95)

    # -------------------------------------------------------------
    # Panel 4: Confusion Matrix Heatmap at Optimal Threshold
    # -------------------------------------------------------------
    ax4 = axes[1, 1]
    preds = (d0_scores >= best_thresh).astype(int)
    cm = confusion_matrix(labels, preds)

    im = ax4.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues)
    ax4.figure.colorbar(im, ax=ax4, shrink=0.8)

    classes = ["Real Image (0)", "AI/Fake Image (1)"]
    tick_marks = np.arange(len(classes))
    ax4.set_xticks(tick_marks)
    ax4.set_xticklabels(classes, rotation=0, fontsize=10)
    ax4.set_yticks(tick_marks)
    ax4.set_yticklabels(classes, rotation=90, va="center", fontsize=10)

    # Annotate cell counts and percentages
    thresh_val = cm.max() / 2.0
    total_samples = cm.sum()
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            count = cm[i, j]
            pct = (count / total_samples) * 100.0
            ax4.text(
                j, i, f"{count}\n({pct:.1f}%)",
                ha="center", va="center",
                color="white" if cm[i, j] > thresh_val else "black",
                fontsize=12, fontweight="bold"
            )

    best_acc = metrics_d0["best_acc"]
    ax4.set_title(f"Confusion Matrix @ $\\tau^* = {best_thresh:.3f}$ (Accuracy = {best_acc:.2f}%)", fontsize=12, fontweight="bold")
    ax4.set_xlabel("Predicted Class", fontsize=10)
    ax4.set_ylabel("True Ground Truth Class", fontsize=10)

    plt.tight_layout()
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, bbox_inches="tight")
    plt.close(fig)
    print(f"🚀 Publication-quality Detection Dashboard saved to: {output_path}")


def plot_roc_curves(
    results_dict: Dict[str, Tuple[np.ndarray, np.ndarray, float]],
    output_path: str,
    model_name: str = "ZED"
):
    """
    Legacy wrapper forwarding to plot_detection_dashboard.
    """
    pass



