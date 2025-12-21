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


class MultiTransformationClassificationDataset(Dataset):
    """
    Dataset that applies multiple transformations in sequence to images.
    Each sample consists of (transformed_image, multi_label).
    
    The model learns to predict which transformations were applied (multi-label classification).
    Ensures no transformation is applied twice to avoid reversible transformation issues.

    Args:
        images_root_path: Directory containing source images
        transformer_keys: List of transformer keys available for selection
        image_size: Target size for images (width, height)
        num_transformations_per_image: Number of transformations to apply in sequence per image
        seed: Random seed for reproducibility
    """

    def __init__(self,
                 images_root_path: Path,
                 transformer_keys: List[str],
                 image_size: Tuple[int, int] = (224, 224),
                 num_transformations_per_image: int = 2,
                 seed: int = 42):
        self.images_root_path = Path(images_root_path)
        self.transformer_keys = transformer_keys
        self.image_size = image_size
        self.num_transformations_per_image = num_transformations_per_image
        self.seed = seed

        # Validate num_transformations
        if num_transformations_per_image < 1 or num_transformations_per_image > len(transformer_keys):
            raise ValueError(
                f"num_transformations_per_image must be between 1 and {len(transformer_keys)}, "
                f"got {num_transformations_per_image}"
            )

        # Set random seed for reproducibility
        random.seed(seed)
        np.random.seed(seed)

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

        # Create samples: each image gets N unique random transformations applied in sequence
        self.samples = []
        for img_path in self.image_paths:
            # Randomly select N unique transformations for this image
            selected_indices = random.sample(range(len(transformer_keys)), num_transformations_per_image)
            self.samples.append((img_path, selected_indices))

        logger.info(
            f"Created {len(self.samples)} samples "
            f"({len(self.image_paths)} images × {num_transformations_per_image} transformations per image)"
        )

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

    def __getitem__(self, idx) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Returns:
            image: Normalized PyTorch tensor (3, H, W)
            label: Multi-hot binary vector indicating which transformations were applied
        """
        img_path, transformer_indices = self.samples[idx]

        # Load image in BGR (OpenCV format)
        image = cv2.imread(str(img_path))
        if image is None:
            raise ValueError(f"Failed to load image: {img_path}")

        # Apply transformations in sequence
        transformed_image = image.copy()
        for transformer_idx in transformer_indices:
            transformer = self.transformers[transformer_idx]
            try:
                transformed_image = transformer.transform(transformed_image)
            except Exception as e:
                logger.error(f"Failed to apply transformer {self.transformer_keys[transformer_idx]} to {img_path}: {e}")
                raise

        # Resize to target size
        transformed_image = cv2.resize(transformed_image, self.image_size)

        # Convert BGR to RGB for PyTorch
        transformed_image_rgb = cv2.cvtColor(transformed_image, cv2.COLOR_BGR2RGB)

        # Normalize and convert to tensor
        image_tensor = self.normalize(transformed_image_rgb)

        # Create multi-hot label vector
        label = torch.zeros(len(self.transformer_keys), dtype=torch.float32)
        for transformer_idx in transformer_indices:
            label[transformer_idx] = 1.0

        return image_tensor, label

    def get_num_classes(self) -> int:
        """Returns number of transformation classes."""
        return len(self.transformers)

    def get_class_names(self) -> List[str]:
        """Returns list of transformation class names."""
        return self.transformer_keys

    def get_sample_info(self, idx: int) -> dict:
        """Get detailed information about a sample."""
        img_path, transformer_indices = self.samples[idx]
        return {
            "image_path": str(img_path),
            "transformer_indices": transformer_indices,
            "transformer_keys": [self.transformer_keys[i] for i in transformer_indices],
            "num_transformations": len(transformer_indices)
        }
