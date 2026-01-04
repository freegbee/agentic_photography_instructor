"""
Rule of Thirds crop transformers.

Crops image to place center of mass on rule-of-thirds gridlines.
The rule of thirds divides an image into 9 equal parts with 2 horizontal and 2 vertical lines.
Key subjects should be placed at the intersection points for better composition.
"""
import cv2
import numpy as np
from numpy import ndarray

from transformer.cropping.abstract_cropping_transformer import AbstractCroppingTransformer


class RuleOfThirdsCropTransformer(AbstractCroppingTransformer):
    """Crop to place image center of mass on rule-of-thirds gridlines."""

    label = "CROP_RULE_OF_THIRDS"
    description = "Crop to place center of mass on rule-of-thirds intersection"

    def transform(self, image: ndarray) -> ndarray:
        """
        Crop image to improve composition using rule of thirds.

        Strategy:
        1. Find the center of mass (brightness-weighted)
        2. Crop to place it near a rule-of-thirds intersection
        3. Maintain at least 75% of original dimensions to avoid excessive quality loss
        """
        height, width = image.shape[:2]

        # Ensure image is large enough (minimum 512x512 for meaningful crops)
        if height < 512 or width < 512:
            return image  # Skip cropping if image too small

        # Convert to grayscale for center of mass calculation
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # Calculate center of mass (brightness-weighted)
        moments = cv2.moments(gray)
        if moments['m00'] != 0:
            cx = int(moments['m10'] / moments['m00'])
            cy = int(moments['m01'] / moments['m00'])
        else:
            # Fallback to image center if moments fail
            cx, cy = width // 2, height // 2

        # Target crop size: 75-85% of original (maintains quality while improving composition)
        crop_ratio = 0.80
        crop_width = int(width * crop_ratio)
        crop_height = int(height * crop_ratio)

        # Rule of thirds positions (1/3 and 2/3 along each axis)
        third_x = crop_width // 3
        third_y = crop_height // 3

        # Determine which rule-of-thirds intersection is closest to center of mass
        # Intersections are at (1/3, 1/3), (1/3, 2/3), (2/3, 1/3), (2/3, 2/3)
        intersections = [
            (third_x, third_y),           # Top-left
            (third_x, 2 * third_y),       # Bottom-left
            (2 * third_x, third_y),       # Top-right
            (2 * third_x, 2 * third_y)    # Bottom-right
        ]

        # Choose intersection based on where center of mass is
        # This places the subject near (but not exactly on) the intersection
        if cx < width // 2 and cy < height // 2:
            target_x, target_y = intersections[0]  # Top-left
        elif cx < width // 2 and cy >= height // 2:
            target_x, target_y = intersections[1]  # Bottom-left
        elif cx >= width // 2 and cy < height // 2:
            target_x, target_y = intersections[2]  # Top-right
        else:
            target_x, target_y = intersections[3]  # Bottom-right

        # Calculate crop rectangle to place center of mass near target intersection
        start_x = max(0, cx - target_x)
        start_y = max(0, cy - target_y)

        # Ensure we don't go out of bounds
        start_x = min(start_x, width - crop_width)
        start_y = min(start_y, height - crop_height)

        # Perform crop
        cropped = image[start_y:start_y + crop_height, start_x:start_x + crop_width]

        return cropped
