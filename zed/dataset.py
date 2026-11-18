"""
Dataset loaders for Real images (Training) and Real/Fake images (Zero-Shot Detection).
Paper: "Zero-Shot Detection of AI-Generated Images" (Cozzolino et al.)
"""

import os
from pathlib import Path
from typing import Tuple, List, Optional, Callable

import torch
from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms as T

import numpy as np

class RealImageDataset(Dataset):
    """
    Dataset loader for REAL images used in training/validating the SReC Density Estimator.
    Recursively scans data_dir and all its subfolders for image files.
    """

    def __init__(
        self,
        data_dir: Optional[str] = None,
        image_paths: Optional[List[Path]] = None,
        image_size: Tuple[int, int] = (256, 256),
        transform: Optional[Callable] = None
    ):
        super().__init__()
        self.image_size = image_size
        
        if image_paths is not None:
            self.image_paths = image_paths
        elif data_dir is not None:
            self.data_dir = Path(data_dir)
            valid_exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}
            # Discover all image files recursively across all subfolders (ignore hidden metadata files starting with '.')
            self.image_paths = sorted([
                p for p in self.data_dir.rglob("*")
                if p.is_file() and not p.name.startswith(".") and p.suffix.lower() in valid_exts
            ])
        else:
            raise ValueError("Either 'data_dir' or 'image_paths' must be provided.")

        if transform is not None:
            self.transform = transform
        else:
            self.transform = T.Compose([
                T.Resize(image_size, interpolation=T.InterpolationMode.BILINEAR),
                T.CenterCrop(image_size),
                T.ToTensor() # Converts PIL [0, 255] to Tensor [0.0, 1.0]
            ])

    def __len__(self) -> int:
        return len(self.image_paths)

    def __getitem__(self, idx: int) -> torch.Tensor:
        path = self.image_paths[idx]
        with Image.open(path) as img:
            image = img.convert("RGB")
        tensor_img = self.transform(image)
        # Convert to range [0, 255] if still in [0.0, 1.0], avoiding accidental double scaling
        if tensor_img.max() <= 1.0 + 1e-5:
            tensor_img = tensor_img * 255.0
        return torch.clamp(tensor_img, 0.0, 255.0)


def create_train_val_datasets(
    data_dir: str,
    val_split: float = 0.15,
    image_size: Tuple[int, int] = (256, 256),
    train_transform: Optional[Callable] = None,
    val_transform: Optional[Callable] = None,
    seed: int = 42
) -> Tuple[RealImageDataset, Optional[RealImageDataset]]:
    """
    Scans data_dir recursively (including all subfolders), collects all image paths,
    and splits them deterministically into Training and Validation RealImageDatasets.
    """
    data_path = Path(data_dir)
    valid_exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}
    
    # Quét đệ quy toàn bộ thư mục và các thư mục con (bỏ qua các file ẩn ._*)
    all_paths = sorted([
        p for p in data_path.rglob("*")
        if p.is_file() and not p.name.startswith(".") and p.suffix.lower() in valid_exts
    ])
    
    total_count = len(all_paths)
    if total_count == 0:
        raise ValueError(f"No valid image files found in '{data_dir}' (scanned recursively in all subfolders).")

    print(f"📁 Recursively scanned '{data_dir}': Found {total_count} images across all subfolders.")

    if train_transform is None:
        train_transform = T.Compose([
            T.Resize(image_size, interpolation=T.InterpolationMode.BILINEAR),
            T.RandomCrop(image_size, pad_if_needed=True),
            T.RandomHorizontalFlip(p=0.5),
            T.ToTensor()
        ])

    if val_transform is None:
        val_transform = T.Compose([
            T.Resize(image_size, interpolation=T.InterpolationMode.BILINEAR),
            T.CenterCrop(image_size),
            T.ToTensor()
        ])

    if val_split <= 0.0 or total_count == 1:
        train_ds = RealImageDataset(image_paths=all_paths, image_size=image_size, transform=train_transform)
        print(f"   -> Training set: {len(train_ds)} images | Validation set: 0 images (val_split=0)")
        return train_ds, None

    # Shuffle deterministically
    rng = np.random.default_rng(seed)
    indices = np.arange(total_count)
    rng.shuffle(indices)

    num_val = max(1, int(total_count * val_split))
    val_indices = indices[:num_val]
    train_indices = indices[num_val:]

    train_paths = [all_paths[i] for i in train_indices]
    val_paths = [all_paths[i] for i in val_indices]

    train_ds = RealImageDataset(image_paths=train_paths, image_size=image_size, transform=train_transform)
    val_ds = RealImageDataset(image_paths=val_paths, image_size=image_size, transform=val_transform)

    print(f"   -> Split: Training set = {len(train_ds)} images | Validation set = {len(val_ds)} images (val_split={val_split:.2f})")
    return train_ds, val_ds



class EvaluationImageDataset(Dataset):
    """
    Dataset loader for Zero-Shot Evaluation (Test set containing Real and Fake images).
    Labels: 0 for REAL images, 1 for FAKE (AI-generated) images.
    Supports either direct directory scanning or pre-split items list.
    """

    def __init__(
        self,
        real_dir: Optional[str] = None,
        fake_dir: Optional[str] = None,
        items: Optional[List[Tuple[Path, int]]] = None,
        image_size: Tuple[int, int] = (256, 256),
        max_samples_per_class: Optional[int] = None,
        max_fake_samples: Optional[int] = None,
        max_real_samples: Optional[int] = None,
        transform: Optional[Callable] = None
    ):
        super().__init__()
        self.image_size = image_size
        
        valid_exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}
        
        if items is not None:
            self.items = items
            self.real_dir = Path(real_dir) if real_dir else None
            self.fake_dir = Path(fake_dir) if fake_dir else None
            self.num_real = sum(1 for _, label in self.items if label == 0)
            self.num_fake = sum(1 for _, label in self.items if label == 1)
        elif real_dir is not None and fake_dir is not None:
            self.real_dir = Path(real_dir)
            self.fake_dir = Path(fake_dir)
            
            effective_real_limit = max_real_samples if max_real_samples is not None else max_samples_per_class
            effective_fake_limit = max_fake_samples if max_fake_samples is not None else max_samples_per_class

            real_items = []
            if self.real_dir.exists():
                for p in self.real_dir.rglob("*"):
                    if p.is_file() and not p.name.startswith(".") and p.suffix.lower() in valid_exts:
                        real_items.append((p, 0)) # Label 0 = Real
                        if effective_real_limit and len(real_items) >= effective_real_limit:
                            break
                        
            fake_items = []
            if self.fake_dir.exists():
                for p in self.fake_dir.rglob("*"):
                    if p.is_file() and not p.name.startswith(".") and p.suffix.lower() in valid_exts:
                        fake_items.append((p, 1)) # Label 1 = Fake
                        if effective_fake_limit and len(fake_items) >= effective_fake_limit:
                            break

            self.num_real = len(real_items)
            self.num_fake = len(fake_items)
            self.items = real_items + fake_items
        else:
            raise ValueError("Either 'items' or both 'real_dir' and 'fake_dir' must be provided.")

        print(f"📊 Evaluation Dataset Loaded: {self.num_real} Real images, {self.num_fake} Fake images (Total: {len(self.items)}).")

        if transform is not None:
            self.transform = transform
        else:
            # Default to zero-blur native center crop preserving raw camera sensor noise
            from .augmentations import NativeCenterCropTransform
            self.transform = NativeCenterCropTransform(image_size=image_size)

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int, str]:
        path, label = self.items[idx]
        with Image.open(path) as img:
            image = img.convert("RGB")
        tensor_img = self.transform(image)
        # Ensure tensor is scaled to [0, 255]
        if tensor_img.max() <= 1.0 + 1e-5:
            tensor_img = tensor_img * 255.0
        return torch.clamp(tensor_img, 0.0, 255.0), label, str(path)


def create_split_evaluation_datasets(
    real_dir: str,
    fake_dir: str,
    test_split: float = 0.30,
    seed: int = 42,
    image_size: Tuple[int, int] = (256, 256),
    max_samples_per_class: Optional[int] = None,
    max_fake_samples: Optional[int] = None,
    max_real_samples: Optional[int] = None,
    transform: Optional[Callable] = None
) -> Tuple[EvaluationImageDataset, Optional[EvaluationImageDataset]]:
    """
    Scans real_dir and fake_dir recursively, collects image paths for Real and Fake,
    and splits both classes deterministically (stratified) into:
      - Calibration Set (1 - test_split, default 70%): used to establish decision thresholds
      - Test Set (test_split, default 30%): used for out-of-sample evaluation

    Args:
        real_dir: Directory containing real images
        fake_dir: Directory containing fake / AI-generated images
        test_split: Proportion of samples reserved for testing (default: 0.30)
        seed: Random seed for reproducible splitting (default: 42)
        image_size: Target image dimensions (default: (256, 256))
        max_samples_per_class: Optional overall cap per class
        max_fake_samples: Optional cap on fake images
        max_real_samples: Optional cap on real images
        transform: Optional image transform callable

    Returns:
        (calib_dataset, test_dataset)
    """
    real_path = Path(real_dir)
    fake_path = Path(fake_dir)
    valid_exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}

    effective_real_limit = max_real_samples if max_real_samples is not None else max_samples_per_class
    effective_fake_limit = max_fake_samples if max_fake_samples is not None else max_samples_per_class

    # 1. Collect all real image paths
    real_items: List[Tuple[Path, int]] = []
    if real_path.exists():
        for p in real_path.rglob("*"):
            if p.is_file() and not p.name.startswith(".") and p.suffix.lower() in valid_exts:
                real_items.append((p, 0))
                if effective_real_limit and len(real_items) >= effective_real_limit:
                    break

    # 2. Collect all fake image paths
    fake_items: List[Tuple[Path, int]] = []
    if fake_path.exists():
        for p in fake_path.rglob("*"):
            if p.is_file() and not p.name.startswith(".") and p.suffix.lower() in valid_exts:
                fake_items.append((p, 1))
                if effective_fake_limit and len(fake_items) >= effective_fake_limit:
                    break

    if len(real_items) == 0 and len(fake_items) == 0:
        raise ValueError(f"No valid images found in '{real_dir}' or '{fake_dir}'.")

    # If test_split <= 0 or single sample, return full dataset as calibration
    if test_split <= 0.0 or (len(real_items) + len(fake_items)) <= 1:
        full_ds = EvaluationImageDataset(
            items=real_items + fake_items,
            real_dir=real_dir,
            fake_dir=fake_dir,
            image_size=image_size,
            transform=transform
        )
        print("💡 test_split <= 0.0: Returning full dataset without split.")
        return full_ds, None

    # Deterministic stratified split
    rng = np.random.default_rng(seed)

    # Split Real images
    num_real = len(real_items)
    if num_real > 1:
        real_idx = np.arange(num_real)
        rng.shuffle(real_idx)
        num_real_test = max(1, int(num_real * test_split))
        real_test = [real_items[i] for i in real_idx[:num_real_test]]
        real_calib = [real_items[i] for i in real_idx[num_real_test:]]
    else:
        real_test = []
        real_calib = list(real_items)

    # Split Fake images (use same RNG stream)
    num_fake = len(fake_items)
    if num_fake > 1:
        fake_idx = np.arange(num_fake)
        rng.shuffle(fake_idx)
        num_fake_test = max(1, int(num_fake * test_split))
        fake_test = [fake_items[i] for i in fake_idx[:num_fake_test]]
        fake_calib = [fake_items[i] for i in fake_idx[num_fake_test:]]
    else:
        fake_test = []
        fake_calib = list(fake_items)

    calib_items = real_calib + fake_calib
    test_items = real_test + fake_test

    print(f"\n📂 Stratified Split (Seed={seed}, Test Split={test_split*100:.1f}%):")
    print(f"   -> Calibration Set ({(1-test_split)*100:.1f}%): {len(real_calib)} Real + {len(fake_calib)} Fake = {len(calib_items)} images")
    print(f"   -> Test Set ({test_split*100:.1f}%): {len(real_test)} Real + {len(fake_test)} Fake = {len(test_items)} images\n")

    calib_ds = EvaluationImageDataset(
        items=calib_items,
        real_dir=real_dir,
        fake_dir=fake_dir,
        image_size=image_size,
        transform=transform
    )
    test_ds = EvaluationImageDataset(
        items=test_items,
        real_dir=real_dir,
        fake_dir=fake_dir,
        image_size=image_size,
        transform=transform
    )

    return calib_ds, test_ds
