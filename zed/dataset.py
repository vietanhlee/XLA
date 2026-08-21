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

class RealImageDataset(Dataset):
    """
    Dataset loader for REAL images used in training the SReC Density Estimator.
    Only real images are required!
    """

    def __init__(
        self,
        data_dir: str,
        image_size: Tuple[int, int] = (256, 256),
        transform: Optional[Callable] = None
    ):
        super().__init__()
        self.data_dir = Path(data_dir)
        self.image_size = image_size
        
        # Discover all image files
        valid_exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}
        self.image_paths = [
            p for p in self.data_dir.rglob("*")
            if p.suffix.lower() in valid_exts
        ]

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
