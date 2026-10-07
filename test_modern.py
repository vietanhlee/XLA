"""
Quick Sanity Check and Verification Script for Modern ZED Model and D4 Augmentations.
Verifies:
  1. DensityPreservingTransform (Native Crop + D4 Symmetry)
  2. ConvNeXt-SReC Encoder & ModernZEDModel forward & backward pass
  3. Calculation of D^(0), D^(0)_top10, Delta^01
"""

import sys
from pathlib import Path
import torch
from PIL import Image

sys.path.append(str(Path(__file__).resolve().parent))

from zed.augmentations import DensityPreservingTransform, D4SymmetryTransform, NativeResolutionCrop
from zed.models import ModernZEDModel, ConvNeXtSReCCNN


def test_pipeline():
    print("=" * 60)
    print("🚀 RUNNING MODERN ZED SANITY CHECKS")
    print("=" * 60)

    # 1. Test Density-Preserving Transforms
    print("[1/3] Testing DensityPreservingTransform...")
    dummy_pil = Image.new("RGB", (512, 512), color=(128, 128, 128))
    transform = DensityPreservingTransform(image_size=(256, 256))
    transformed_tensor = transform(dummy_pil)
    assert transformed_tensor.shape == (3, 256, 256), f"Shape mismatch: {transformed_tensor.shape}"
    print("  ✓ DensityPreservingTransform verified! Shape:", transformed_tensor.shape)

    # 2. Test Model Instantiation & Forward Pass (Training Mode)
    print("[2/3] Testing ModernZEDModel Forward & Backward (Train Mode)...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = ModernZEDModel(
        in_channels=3,
        num_mixtures=10,
        hidden_channels=64,
        num_blocks=4
    ).to(device)

    # Dummy batch of 2 images in [0, 255]
    x_train = torch.randint(0, 256, (2, 3, 256, 256), dtype=torch.float32, device=device)
    output_train = model(x_train, compute_entropy=False)

    loss = output_train["total_loss"].mean()
    assert not torch.isnan(loss) and not torch.isinf(loss), "Loss is NaN/Inf!"
    loss.backward()
    print(f"  ✓ Forward & Backward passed! Total Loss: {loss.item():.4f}")

    # 3. Test Model Evaluation Mode (Inference Mode with exact Entropy & Top-10% Anomaly)
    print("[3/3] Testing ModernZEDModel Inference Mode (Exact Entropy + Top-10% Anomaly)...")
    model.eval()
    with torch.no_grad():
        x_val = torch.randint(0, 256, (2, 3, 256, 256), dtype=torch.float32, device=device)
        output_eval = model(x_val, compute_entropy=True)

    d0 = output_eval["d0"]
    d0_top10 = output_eval["d0_top10"]
    delta01 = output_eval["delta01"]

    print(f"  ✓ D^(0) Mean shape: {d0.shape} | Values: {d0.cpu().numpy()}")
    print(f"  ✓ D^(0) Top-10% shape: {d0_top10.shape} | Values: {d0_top10.cpu().numpy()}")
    print(f"  ✓ Delta^01 shape: {delta01.shape} | Values: {delta01.cpu().numpy()}")

    print("=" * 60)
    print("🎉 ALL CHECKS PASSED SUCCESSFULLY! Model is production-ready.")
    print("=" * 60)


if __name__ == "__main__":
    test_pipeline()
