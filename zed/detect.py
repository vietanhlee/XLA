"""
Zero-Shot AI Image Detection & Evaluation Script.
Paper: "Zero-Shot Detection of AI-Generated Images" (Cozzolino et al.)

Evaluates test images (Real vs Fake/AI-generated) using the learned real-image density model.
Calculates decision statistics D^(0) and |Delta^01|, and outputs ROC-AUC & Detection Accuracy.
"""

import os
import sys
import argparse
from pathlib import Path

from typing import Tuple
import torch
import numpy as np
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

# Add parent dir to path
sys.path.append(str(Path(__file__).resolve().parent.parent))

from config import ModelConfig, DetectConfig
from zed.models.zed_model import ZEDModel
from zed.dataset import EvaluationImageDataset
import json
from zed.utils import load_checkpoint, compute_metrics, plot_detection_dashboard, get_safe_device, wrap_model_multigpu

def evaluate_zero_shot(
    model: ZEDModel,
    dataloader: DataLoader,
    device: torch.device
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Evaluates test set images and extracts decision statistics.
    
    Returns:
        labels: True binary labels (0 = Real, 1 = Fake)
        d0_scores: D^(0) coding cost gap
        abs_d0_scores: |D^(0)|
        delta01_scores: |Delta^01| = |D^(0) - D^(1)|
    """
    model.eval()
    
    all_labels = []
    all_d0 = []
    all_abs_d0 = []
    all_delta01 = []

    print("Evaluating test images (computing NLL & exact Entropy)...")
    with torch.no_grad():
        for images, labels, paths in tqdm(dataloader, desc="Detecting"):
            images = images.to(device)
            
            # Forward pass computing exact NLL and Entropy
            output_dict = model(images, compute_entropy=True)
            
            d0 = output_dict["d0"].cpu().numpy()
            abs_d0 = output_dict["abs_d0"].cpu().numpy()
            abs_delta01 = output_dict["abs_delta01"].cpu().numpy()

            all_labels.extend(labels.numpy())
            all_d0.extend(d0)
            all_abs_d0.extend(abs_d0)
            all_delta01.extend(abs_delta01)

    return (
        np.array(all_labels),
        np.array(all_d0),
        np.array(all_abs_d0),
        np.array(all_delta01)
    )

def main():
    parser = argparse.ArgumentParser(description="Zero-Shot Detection of AI-Generated Images")
    parser.add_argument("--checkpoint", type=str, default="checkpoints/zed_best.pth", help="Path to pre-trained ZED checkpoint.")
    parser.add_argument("--real_dir", type=str, default="data/test/real", help="Directory containing test REAL images.")
    parser.add_argument("--fake_dir", type=str, default="data/test/fake", help="Directory containing test FAKE / AI images.")
    parser.add_argument("--batch_size", type=int, default=4, help="Batch size for inference.")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu", help="Device (cuda/cpu).")
    parser.add_argument("--multigpu", action="store_true", default=True, help="Enable multi-GPU DataParallel inference if multiple GPUs exist (default: True).")
    parser.add_argument("--no_multigpu", dest="multigpu", action="store_false", help="Force single-GPU mode even if multiple GPUs exist.")
    args = parser.parse_args()

    device = get_safe_device(args.device)
    print(f"=== ZED Zero-Shot Detection Evaluation ===")
    print(f"Device: {device}")
    print(f"Real Dir: {args.real_dir}")
    print(f"Fake Dir: {args.fake_dir}")

    # Load Model
    model_cfg = ModelConfig()
    model = ZEDModel(
        in_channels=model_cfg.in_channels,
        num_mixtures=model_cfg.num_mixtures,
        hidden_channels=model_cfg.hidden_channels,
        num_res_blocks=model_cfg.num_res_blocks,
        min_log_scale=model_cfg.min_log_scale
    ).to(device)

    if os.path.exists(args.checkpoint):
        load_checkpoint(args.checkpoint, model, device=device.type)
    else:
        print(f"Warning: Checkpoint '{args.checkpoint}' not found! Running evaluation with randomly initialized model.")

    # Multi-GPU Auto Detection & DataParallel Wrap
    model, gpu_count = wrap_model_multigpu(model, device, use_multigpu=args.multigpu)

    # Load Dataset
    dataset = EvaluationImageDataset(
        real_dir=args.real_dir,
        fake_dir=args.fake_dir
    )

    if len(dataset) == 0:
        print(f"ERROR: No evaluation images found in '{args.real_dir}' or '{args.fake_dir}'.")
        return

    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=2 if device.type == "cuda" else 0
    )

    # Evaluate
    labels, d0_scores, abs_d0_scores, delta01_scores = evaluate_zero_shot(model, dataloader, device)

    # Compute Metrics
    print("\n" + "="*50)
    print(" === ZERO-SHOT DETECTION PERFORMANCE RESULTS ===")
    print("="*50)

    # Statistic 1: D^(0)
    metrics_d0 = compute_metrics(d0_scores, labels)
    print(f"[Stat D^(0)] -> ROC-AUC: {metrics_d0['auc']:.2f}% | Best Acc: {metrics_d0['best_acc']:.2f}% | Threshold: {metrics_d0['best_threshold']:.4f}")

    # Statistic 2: |D^(0)|
    metrics_abs_d0 = compute_metrics(abs_d0_scores, labels)
    print(f"[Stat |D^(0)|] -> ROC-AUC: {metrics_abs_d0['auc']:.2f}% | Best Acc: {metrics_abs_d0['best_acc']:.2f}% | Threshold: {metrics_abs_d0['best_threshold']:.4f}")

    # Statistic 3: |Delta^01|
    metrics_delta01 = compute_metrics(delta01_scores, labels)
    print(f"[Stat |Delta^01|] -> ROC-AUC: {metrics_delta01['auc']:.2f}% | Best Acc: {metrics_delta01['best_acc']:.2f}% | Threshold: {metrics_delta01['best_threshold']:.4f}")

    print("="*50)

    # Save Detection Dashboard Plot and Metrics JSON
    results_dir = Path("results")
    results_dir.mkdir(parents=True, exist_ok=True)
    
    dashboard_path = str(results_dir / "detection_dashboard_standard_zed.png")
    plot_detection_dashboard(
        d0_scores=d0_scores,
        abs_d0_scores=abs_d0_scores,
        delta01_scores=delta01_scores,
        labels=labels,
        metrics_d0=metrics_d0,
        output_path=dashboard_path,
        model_name="Standard ZED"
    )

    summary_metrics = {
        "d0": metrics_d0,
        "abs_d0": metrics_abs_d0,
        "delta01": metrics_delta01
    }
    metrics_json_path = results_dir / "detection_metrics_standard_zed.json"
    with open(metrics_json_path, "w") as f:
        json.dump(summary_metrics, f, indent=2)

    print(f"📊 Saved 4-panel Detection Dashboard plot: {dashboard_path}")
    print(f"📄 Saved metrics JSON: {metrics_json_path}")

if __name__ == "__main__":
    main()
