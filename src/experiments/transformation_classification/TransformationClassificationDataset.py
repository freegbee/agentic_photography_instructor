import logging
from pathlib import Path
from typing import List, Tuple
import random

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision import transforms

from transformer.AbstractTransformer import AbstractTransformer
from utils.Registries import TRANSFORMER_REGISTRY

logger = logging.getLogger(__name__)


class TransformationClassificationDataset(Dataset):
    """
    Dataset that applies transformations to images and creates labeled samples.
    Each sample consists of (transformed_image, transformation_label).

    Args:
        images_root_path: Directory containing source images
        transformer_keys: List of transformer keys (labels) to use
        image_size: Target size for images (width, height)
        seed: Random seed for reproducibility
    """

    def __init__(self,
                 images_root_path: Path,
                 transformer_keys: List[str],
                 image_size: Tuple[int, int] = (224, 224),
                 seed: int = 42):
        self.images_root_path = Path(images_root_path)
        self.transformer_keys = transformer_keys
        self.image_size = image_size
        self.seed = seed

        # Load transformers from registry
        self.transformers: List[AbstractTransformer] = []
        self.transformer_label_to_idx = {}
        for idx, key in enumerate(transformer_keys):
            if key not in TRANSFORMER_REGISTRY:
                raise ValueError(f"Transformer with key '{key}' not found in registry")
            transformer = TRANSFORMER_REGISTRY.get(key)
            self.transformers.append(transformer)
            self.transformer_label_to_idx[key] = idx
            logger.info(f"Loaded transformer {idx}: {key} - {transformer.description}")

        # Find all image files
        self.image_paths = self._find_images(self.images_root_path)
        logger.info(f"Found {len(self.image_paths)} images in {images_root_path}")

        if len(self.image_paths) == 0:
            raise ValueError(f"No images found in {images_root_path}")

        # Create samples: each image gets each transformation
        self.samples = []
        for img_path in self.image_paths:
            for transformer_idx, transformer_key in enumerate(transformer_keys):
                self.samples.append((img_path, transformer_idx, transformer_key))

        logger.info(f"Created {len(self.samples)} samples ({len(self.image_paths)} images × {len(transformer_keys)} transformations)")

        # PyTorch transforms for normalization (ImageNet stats)
        self.normalize = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                               std=[0.229, 0.224, 0.225])
        ])

    def _find_images(self, path: Path) -> List[Path]:
        """Find all image files in directory."""
        image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif'}
        image_paths = []

        for ext in image_extensions:
            image_paths.extend(path.rglob(f'*{ext}'))
            image_paths.extend(path.rglob(f'*{ext.upper()}'))

        return sorted(image_paths)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx) -> Tuple[torch.Tensor, int]:
        """
        Returns:
            image: Normalized PyTorch tensor (3, H, W)
            label: Transformation class index
        """
        img_path, transformer_idx, transformer_key = self.samples[idx]

        # Load image in BGR (OpenCV format)
        image = cv2.imread(str(img_path))
        if image is None:
            raise ValueError(f"Failed to load image: {img_path}")

        # Apply transformation
        transformer = self.transformers[transformer_idx]
        try:
            transformed_image = transformer.transform(image)
        except Exception as e:
            logger.error(f"Failed to apply transformer {transformer_key} to {img_path}: {e}")
            raise

        # Resize to target size
        transformed_image = cv2.resize(transformed_image, self.image_size)

        # Convert BGR to RGB for PyTorch
        transformed_image_rgb = cv2.cvtColor(transformed_image, cv2.COLOR_BGR2RGB)

        # Normalize and convert to tensor
        image_tensor = self.normalize(transformed_image_rgb)

        return image_tensor, transformer_idx

    def get_num_classes(self) -> int:
        """Returns number of transformation classes."""
        return len(self.transformers)

    def get_class_names(self) -> List[str]:
        """Returns list of transformation class names."""
        return self.transformer_keys
