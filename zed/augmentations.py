"""
Robust Data Augmentation Pipeline for Real-Image Density Estimation.
Simulates real-world internet noise (JPEG compression, resizing, sensor noise, blur)
to prevent false positive alarms on compressed/processed real images.
"""

import io
import random
import torch
import torch.nn as nn
import torchvision.transforms as T
import torchvision.transforms.functional as TF
from PIL import Image
from typing import Tuple

class DynamicJPEGCompression:
    """Simulates random JPEG compression artifacting on PIL Image."""
    def __init__(self, quality_range: Tuple[int, int] = (50, 95), p: float = 0.5):
        self.quality_range = quality_range
        self.p = p

    def __call__(self, img: Image.Image) -> Image.Image:
        if random.random() > self.p:
            return img

        quality = random.randint(self.quality_range[0], self.quality_range[1])
        output = io.BytesIO()
        img.save(output, format="JPEG", quality=quality)
        output.seek(0)
        return Image.open(output).convert("RGB")

class AdditiveGaussianNoise:
    """Adds random Gaussian sensor noise to image tensor in range [0.0, 1.0]."""
    def __init__(self, std_range: Tuple[float, float] = (0.0, 0.03), p: float = 0.3):
        self.std_range = std_range
        self.p = p

    def __call__(self, tensor: torch.Tensor) -> torch.Tensor:
        if random.random() > self.p:
            return tensor
        
        std = random.uniform(self.std_range[0], self.std_range[1])
        noise = torch.randn_like(tensor) * std
        return torch.clamp(tensor + noise, 0.0, 1.0)

class RobustRealImageTransform:
    """
    Complete Robust Pipeline for Training Real Image Density Estimator.
    Combines:
      - Random JPEG Compression
      - Random Resizing & Scale Jittering
      - Random Blur
      - Additive Noise
    """

    def __init__(
        self,
        image_size: Tuple[int, int] = (256, 256),
        jpeg_p: float = 0.5,
        jpeg_quality: Tuple[int, int] = (50, 95),
        noise_p: float = 0.3
    ):
        self.image_size = image_size
        self.jpeg_transform = DynamicJPEGCompression(quality_range=jpeg_quality, p=jpeg_p)
        self.noise_transform = AdditiveGaussianNoise(p=noise_p)

        self.pil_transform = T.Compose([
            T.Resize((int(image_size[0] * 1.1), int(image_size[1] * 1.1)), interpolation=T.InterpolationMode.BILINEAR),
            T.RandomCrop(image_size),
        ])

    def __call__(self, img: Image.Image) -> torch.Tensor:
        # 1. Apply PIL transforms & JPEG compression
        img = self.pil_transform(img)
        img = self.jpeg_transform(img)
        
        # 2. Convert to tensor [0.0, 1.0]
        tensor = TF.to_tensor(img)
        
        # 3. Apply tensor noise
        tensor = self.noise_transform(tensor)
        
        # 4. Scale to [0, 255] for 8-bit discrete logistic mixture model
        return tensor * 255.0
