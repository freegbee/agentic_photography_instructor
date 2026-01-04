"""
Aspect ratio optimization transformers.

Crops images to standard aspect ratios for optimal composition.
Different aspect ratios suit different content:
- 16:9: Cinematic, landscapes, wide scenes
- 4:3: Classic photography, balanced composition
- 3:2: 35mm film standard, DSLR default
- 4:5: Portrait orientation, Instagram vertical
- 5:4: Large format photography
- 1:1: Social media, square composition
"""
import cv2
import numpy as np
from numpy import ndarray

from transformer.cropping.abstract_cropping_transformer import AbstractCroppingTransformer


class AspectRatio16x9CropTransformer(AbstractCroppingTransformer):
    """Crop to 16:9 aspect ratio (cinematic/widescreen format)."""

    label = "CROP_16x9"
    description = "Crop to 16:9 aspect ratio for cinematic composition"

    def transform(self, image: ndarray) -> ndarray:
        """
        Crop to 16:9 aspect ratio, centered on image center.

        Maintains maximum dimensions while achieving target aspect ratio.
        """
        height, width = image.shape[:2]

        # Skip if image too small
        if height < 432 or width < 768:  # Minimum for 16:9 at decent quality
            return image

        # Calculate target dimensions for 16:9
        target_ratio = 16.0 / 9.0
        current_ratio = width / height

        if abs(current_ratio - target_ratio) < 0.01:
            return image  # Already close to 16:9

        if current_ratio > target_ratio:
            # Image is too wide, crop width
            new_width = int(height * target_ratio)
            new_height = height
        else:
            # Image is too tall, crop height
            new_width = width
            new_height = int(width / target_ratio)

        # Center crop
        start_x = (width - new_width) // 2
        start_y = (height - new_height) // 2

        return image[start_y:start_y + new_height, start_x:start_x + new_width]


class AspectRatio4x3CropTransformer(AbstractCroppingTransformer):
    """Crop to 4:3 aspect ratio (classic photography format)."""

    label = "CROP_4x3"
    description = "Crop to 4:3 aspect ratio for classic composition"

    def transform(self, image: ndarray) -> ndarray:
        """
        Crop to 4:3 aspect ratio, centered on image center.

        4:3 is a classic photography ratio that works well for most content.
        """
        height, width = image.shape[:2]

        # Skip if image too small
        if height < 384 or width < 512:
            return image

        # Calculate target dimensions for 4:3
        target_ratio = 4.0 / 3.0
        current_ratio = width / height

        if abs(current_ratio - target_ratio) < 0.01:
            return image  # Already close to 4:3

        if current_ratio > target_ratio:
            # Image is too wide, crop width
            new_width = int(height * target_ratio)
            new_height = height
        else:
            # Image is too tall, crop height
            new_width = width
            new_height = int(width / target_ratio)

        # Center crop
        start_x = (width - new_width) // 2
        start_y = (height - new_height) // 2

        return image[start_y:start_y + new_height, start_x:start_x + new_width]


class AspectRatio3x2CropTransformer(AbstractCroppingTransformer):
    """Crop to 3:2 aspect ratio (35mm film standard, DSLR default)."""

    label = "CROP_3x2"
    description = "Crop to 3:2 aspect ratio for DSLR standard composition"

    def transform(self, image: ndarray) -> ndarray:
        """
        Crop to 3:2 aspect ratio, centered on image center.

        3:2 is the standard aspect ratio for 35mm film and most DSLR cameras.
        """
        height, width = image.shape[:2]

        # Skip if image too small
        if height < 384 or width < 576:
            return image

        # Calculate target dimensions for 3:2
        target_ratio = 3.0 / 2.0
        current_ratio = width / height

        if abs(current_ratio - target_ratio) < 0.01:
            return image  # Already close to 3:2

        if current_ratio > target_ratio:
            # Image is too wide, crop width
            new_width = int(height * target_ratio)
            new_height = height
        else:
            # Image is too tall, crop height
            new_width = width
            new_height = int(width / target_ratio)

        # Center crop
        start_x = (width - new_width) // 2
        start_y = (height - new_height) // 2

        return image[start_y:start_y + new_height, start_x:start_x + new_width]


class AspectRatio4x5CropTransformer(AbstractCroppingTransformer):
    """Crop to 4:5 aspect ratio (portrait orientation, Instagram vertical)."""

    label = "CROP_4x5"
    description = "Crop to 4:5 aspect ratio for portrait orientation"

    def transform(self, image: ndarray) -> ndarray:
        """
        Crop to 4:5 aspect ratio, centered on image center.

        4:5 is ideal for portrait orientation and Instagram vertical posts.
        """
        height, width = image.shape[:2]

        # Skip if image too small
        if height < 480 or width < 384:
            return image

        # Calculate target dimensions for 4:5
        target_ratio = 4.0 / 5.0
        current_ratio = width / height

        if abs(current_ratio - target_ratio) < 0.01:
            return image  # Already close to 4:5

        if current_ratio > target_ratio:
            # Image is too wide, crop width
            new_width = int(height * target_ratio)
            new_height = height
        else:
            # Image is too tall, crop height
            new_width = width
            new_height = int(width / target_ratio)

        # Center crop
        start_x = (width - new_width) // 2
        start_y = (height - new_height) // 2

        return image[start_y:start_y + new_height, start_x:start_x + new_width]


class AspectRatio5x4CropTransformer(AbstractCroppingTransformer):
    """Crop to 5:4 aspect ratio (large format photography)."""

    label = "CROP_5x4"
    description = "Crop to 5:4 aspect ratio for large format composition"

    def transform(self, image: ndarray) -> ndarray:
        """
        Crop to 5:4 aspect ratio, centered on image center.

        5:4 is used in large format photography and provides a slightly wider composition than 4:3.
        """
        height, width = image.shape[:2]

        # Skip if image too small
        if height < 384 or width < 480:
            return image

        # Calculate target dimensions for 5:4
        target_ratio = 5.0 / 4.0
        current_ratio = width / height

        if abs(current_ratio - target_ratio) < 0.01:
            return image  # Already close to 5:4

        if current_ratio > target_ratio:
            # Image is too wide, crop width
            new_width = int(height * target_ratio)
            new_height = height
        else:
            # Image is too tall, crop height
            new_width = width
            new_height = int(width / target_ratio)

        # Center crop
        start_x = (width - new_width) // 2
        start_y = (height - new_height) // 2

        return image[start_y:start_y + new_height, start_x:start_x + new_width]


class AspectRatio1x1CropTransformer(AbstractCroppingTransformer):
    """Crop to 1:1 aspect ratio (square format for social media)."""

    label = "CROP_1x1"
    description = "Crop to 1:1 square aspect ratio for social media"

    def transform(self, image: ndarray) -> ndarray:
        """
        Crop to 1:1 square aspect ratio, centered on image center.

        Square format is ideal for Instagram and other social media platforms.
        """
        height, width = image.shape[:2]

        # Skip if image too small
        if height < 512 or width < 512:
            return image

        # Use the smaller dimension as the crop size
        crop_size = min(height, width)

        # Center crop
        start_x = (width - crop_size) // 2
        start_y = (height - crop_size) // 2

        return image[start_y:start_y + crop_size, start_x:start_x + crop_size]
