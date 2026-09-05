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
            # Discover all image files recursively across all subfolders
            self.image_paths = sorted([
                p for p in self.data_dir.rglob("*")
                if p.is_file() and p.suffix.lower() in valid_exts
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
        image = Image.open(path).convert("RGB")
        tensor_img = self.transform(image) # Range [0.0, 1.0]
        # Convert to range [0, 255] for 8-bit discrete logistic mixture model
        return tensor_img * 255.0


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
    
    # Quét đệ quy toàn bộ thư mục và các thư mục con
    all_paths = sorted([
        p for p in data_path.rglob("*")
        if p.is_file() and p.suffix.lower() in valid_exts
    ])
    
    total_count = len(all_paths)
    if total_count == 0:
        raise ValueError(f"No valid image files found in '{data_dir}' (scanned recursively in all subfolders).")

    print(f"📁 Recursively scanned '{data_dir}': Found {total_count} images across all subfolders.")

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
    """

    def __init__(
        self,
        real_dir: str,
        fake_dir: str,
        image_size: Tuple[int, int] = (256, 256)
    ):
        super().__init__()
        self.real_dir = Path(real_dir)
        self.fake_dir = Path(fake_dir)
        self.image_size = image_size
        
        valid_exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}
        
        self.items = []
        if self.real_dir.exists():
            for p in self.real_dir.rglob("*"):
                if p.suffix.lower() in valid_exts:
                    self.items.append((p, 0)) # Label 0 = Real
                    
        if self.fake_dir.exists():
            for p in self.fake_dir.rglob("*"):
                if p.suffix.lower() in valid_exts:
                    self.items.append((p, 1)) # Label 1 = Fake

        self.transform = T.Compose([
            T.Resize(image_size, interpolation=T.InterpolationMode.BILINEAR),
            T.CenterCrop(image_size),
            T.ToTensor()
        ])

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int, str]:
        path, label = self.items[idx]
        image = Image.open(path).convert("RGB")
        tensor_img = self.transform(image) * 255.0
        return tensor_img, label, str(path)
