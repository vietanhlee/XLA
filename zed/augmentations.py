"""
Density-Preserving Data Augmentation Pipeline for Real-Image Density Estimation.
Guarantees:
  1. Zero resampling blur: NativeResolutionCrop preserves raw camera sensor PRNU & Bayer CFA.
  2. Perfect pixel integrity: D4 discrete symmetries (Rot90 + Flips) without interpolation.
  3. Clean target distribution: Eliminates artificial noise injection that skews NLL.
"""

import random
from typing import Tuple
from PIL import Image

import torch
import torchvision.transforms.functional as TF


class NativeResolutionCrop:
    """
    Directly crops a patch of `image_size` from native resolution without bilinear resampling.
    Preserves 100% of the raw camera sensor PRNU noise and Bayer demosaicing traces.
    Only resizes if image dimensions are smaller than image_size.
    """
    def __init__(self, image_size: Tuple[int, int] = (256, 256)):
        self.image_size = image_size

    def __call__(self, img: Image.Image) -> Image.Image:
        w, h = img.size
        target_h, target_w = self.image_size

        if w < target_w or h < target_h:
            scale = max(target_w / w, target_h / h)
            new_w = int(w * scale + 0.5)
            new_h = int(h * scale + 0.5)
            img = img.resize((new_w, new_h), resample=Image.Resampling.BILINEAR)
            w, h = img.size

        # Random crop without resampling
        left = random.randint(0, max(0, w - target_w))
        top = random.randint(0, max(0, h - target_h))
        return img.crop((left, top, left + target_w, top + target_h))


class D4SymmetryTransform:
    """
    Applies random isometric symmetry from the Dihedral group D4 (8 symmetries):
    - Random 90, 180, 270 degree rotation
    - Random Horizontal and Vertical Flip
    Preserves exact integer pixel values with zero interpolation / blur artifacts.
    """
    def __init__(self, p_flip: float = 0.5):
        self.p_flip = p_flip

    def __call__(self, img: Image.Image) -> Image.Image:
        # 1. Random 90-degree discrete rotation
        angle = random.choice([0, 90, 180, 270])
        if angle == 90:
            img = img.transpose(Image.Transpose.ROTATE_90)
        elif angle == 180:
            img = img.transpose(Image.Transpose.ROTATE_180)
        elif angle == 270:
            img = img.transpose(Image.Transpose.ROTATE_270)

        # 2. Random horizontal flip
        if random.random() < self.p_flip:
            img = img.transpose(Image.Transpose.FLIP_LEFT_RIGHT)

        # 3. Random vertical flip
        if random.random() < self.p_flip:
            img = img.transpose(Image.Transpose.FLIP_TOP_BOTTOM)

        return img


class DensityPreservingTransform:
    """
    State-of-the-Art Data Transform for Training Likelihood-based Density Estimators (ZED).
    Applies Native Crop followed by discrete D4 symmetries.
    """
    def __init__(self, image_size: Tuple[int, int] = (256, 256)):
        self.crop = NativeResolutionCrop(image_size=image_size)
        self.d4 = D4SymmetryTransform()

    def __call__(self, img: Image.Image) -> torch.Tensor:
        # 1. Native crop
        img = self.crop(img)
        # 2. D4 symmetry
        img = self.d4(img)
        # 3. Convert to float tensor [0.0, 1.0]
        return TF.to_tensor(img)


class NativeCenterCropTransform:
    """
    Deterministic Center Crop at native camera resolution for Validation & Evaluation.
    Zero-interpolation: Crops the central image_size patch directly without bilinear resampling.
    Preserves 100% of raw camera PRNU noise and AI generation artifacts on test sets.
    Only resizes if image dimensions are smaller than image_size.
    """
    def __init__(self, image_size: Tuple[int, int] = (256, 256)):
        self.image_size = image_size

    def __call__(self, img: Image.Image) -> torch.Tensor:
        w, h = img.size
        target_h, target_w = self.image_size

        if w < target_w or h < target_h:
            scale = max(target_w / w, target_h / h)
            new_w = int(w * scale + 0.5)
            new_h = int(h * scale + 0.5)
            img = img.resize((new_w, new_h), resample=Image.Resampling.BILINEAR)
            w, h = img.size

        # Center crop without resampling
        left = max(0, (w - target_w) // 2)
        top = max(0, (h - target_h) // 2)
        cropped_img = img.crop((left, top, left + target_w, top + target_h))
        return TF.to_tensor(cropped_img)
