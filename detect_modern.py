"""
Zero-Shot AI Image Detection & Evaluation for Modern ZED Model.
Evaluates test images using ModernZEDModel (ConvNeXt-SReC + Top-10% Patch Anomaly).
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

sys.path.append(str(Path(__file__).resolve().parent))

import json
from config import ModelConfig
from zed.models import ModernZEDModel
from zed.dataset import EvaluationImageDataset
from zed.utils import load_checkpoint, compute_metrics, plot_detection_dashboard, get_safe_device, wrap_model_multigpu


def evaluate_modern_zero_shot(
    model: ModernZEDModel,
    dataloader: DataLoader,
    device: torch.device
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    model.eval()

    all_labels = []
    all_d0 = []
    all_abs_d0 = []
    all_d0_top10 = []
    all_delta01 = []

    print("Evaluating test images with Modern ZED (computing exact NLL, Entropy, and Patch Anomaly)...")
    with torch.no_grad():
        for images, labels, paths in tqdm(dataloader, desc="Detecting Modern"):
            images = images.to(device)
            output_dict = model(images, compute_entropy=True)

            d0 = output_dict["d0"].cpu().numpy()
            abs_d0 = output_dict["abs_d0"].cpu().numpy()
            d0_top10 = output_dict["d0_top10"].cpu().numpy()
            abs_delta01 = output_dict["abs_delta01"].cpu().numpy()

            all_labels.extend(labels.numpy())
            all_d0.extend(d0)
            all_abs_d0.extend(abs_d0)
            all_d0_top10.extend(d0_top10)
            all_delta01.extend(abs_delta01)

    return (
        np.array(all_labels),
        np.array(all_d0),
        np.array(all_abs_d0),
        np.array(all_d0_top10),
        np.array(all_delta01)
    )


def main():
    parser = argparse.ArgumentParser(description="Zero-Shot Detection with Modern ZED Model (ConvNeXt-SReC)")
    parser.add_argument("--checkpoint", type=str, default="checkpoints/zed_modern_best.pth", help="Path to Modern ZED checkpoint.")
    parser.add_argument("--real_dir", type=str, default="data/test/real", help="Directory containing test REAL images.")
    parser.add_argument("--fake_dir", type=str, default="data/test/fake", help="Directory containing test FAKE / AI images.")
    parser.add_argument("--batch_size", type=int, default=4, help="Batch size for inference.")
    parser.add_argument("--max_samples", type=int, default=None, help="Max test images per class (real/fake). E.g. 1000 for fast testing.")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu", help="Device (cuda/cpu).")
    parser.add_argument("--multigpu", action="store_true", default=True, help="Enable multi-GPU DataParallel inference if multiple GPUs exist (default: True).")
    parser.add_argument("--no_multigpu", dest="multigpu", action="store_false", help="Force single-GPU mode even if multiple GPUs exist.")
    args = parser.parse_args()

    device = get_safe_device(args.device)
    print("=== Modern ZED Zero-Shot Detection Evaluation ===")
    print("Architecture: ConvNeXt-SReC (7x7 Depthwise Conv + ECA)")
    print(f"Device: {device}")
    if args.max_samples:
        print(f"Fast Evaluation Mode: Max {args.max_samples} images per class.")

    model_cfg = ModelConfig()
    model = ModernZEDModel(
        in_channels=model_cfg.in_channels,
        num_mixtures=model_cfg.num_mixtures,
        hidden_channels=model_cfg.hidden_channels,
        num_blocks=model_cfg.num_res_blocks,
        min_log_scale=model_cfg.min_log_scale
    ).to(device)

    if os.path.exists(args.checkpoint):
        load_checkpoint(args.checkpoint, model, device=device.type)
    else:
        print(f"Warning: Checkpoint '{args.checkpoint}' not found! Running evaluation with randomly initialized Modern model.")

    # Multi-GPU Auto Detection & DataParallel Wrap
    model, gpu_count = wrap_model_multigpu(model, device, use_multigpu=args.multigpu)

    dataset = EvaluationImageDataset(
        real_dir=args.real_dir,
        fake_dir=args.fake_dir,
        max_samples_per_class=args.max_samples
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

    labels, d0_scores, abs_d0_scores, d0_top10_scores, delta01_scores = evaluate_modern_zero_shot(model, dataloader, device)

    print("\n" + "="*60)
    print(" === MODERN ZED ZERO-SHOT DETECTION PERFORMANCE RESULTS ===")
    print("="*60)

    metrics_d0 = compute_metrics(d0_scores, labels)
    print(f"[Stat D^(0) Mean]       -> ROC-AUC: {metrics_d0['auc']:.2f}% | Best Acc: {metrics_d0['best_acc']:.2f}% | Threshold: {metrics_d0['best_threshold']:.4f}")

    metrics_d0_top10 = compute_metrics(d0_top10_scores, labels)
    print(f"[Stat D^(0) Top-10%]    -> ROC-AUC: {metrics_d0_top10['auc']:.2f}% | Best Acc: {metrics_d0_top10['best_acc']:.2f}% | Threshold: {metrics_d0_top10['best_threshold']:.4f}")

    metrics_abs_d0 = compute_metrics(abs_d0_scores, labels)
    print(f"[Stat |D^(0)|]          -> ROC-AUC: {metrics_abs_d0['auc']:.2f}% | Best Acc: {metrics_abs_d0['best_acc']:.2f}% | Threshold: {metrics_abs_d0['best_threshold']:.4f}")

    metrics_delta01 = compute_metrics(delta01_scores, labels)
    print(f"[Stat |Delta^01|]       -> ROC-AUC: {metrics_delta01['auc']:.2f}% | Best Acc: {metrics_delta01['best_acc']:.2f}% | Threshold: {metrics_delta01['best_threshold']:.4f}")

    print("="*60)

    # Save Detection Dashboard Plot and Metrics JSON
    results_dir = Path("results")
    results_dir.mkdir(parents=True, exist_ok=True)

    dashboard_path = str(results_dir / "detection_dashboard_modern_zed.png")
    plot_detection_dashboard(
        d0_scores=d0_scores,
        abs_d0_scores=abs_d0_scores,
        delta01_scores=delta01_scores,
        labels=labels,
        metrics_d0=metrics_d0,
        output_path=dashboard_path,
        model_name="Modern ZED",
        d0_top10_scores=d0_top10_scores
    )

    summary_metrics = {
        "d0_mean": metrics_d0,
        "d0_top10": metrics_d0_top10,
        "abs_d0": metrics_abs_d0,
        "delta01": metrics_delta01
    }
    metrics_json_path = results_dir / "detection_metrics_modern_zed.json"
    with open(metrics_json_path, "w") as f:
        json.dump(summary_metrics, f, indent=2)

    print(f"📊 Saved 4-panel Modern Detection Dashboard plot: {dashboard_path}")
    print(f"📄 Saved Modern metrics JSON: {metrics_json_path}")


if __name__ == "__main__":
    main()
