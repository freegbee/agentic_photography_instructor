"""
Horizon leveling transformer.

Detects and straightens tilted horizons in landscape images.
Uses Hough Line Transform to detect horizontal lines and rotates the image to level them.
"""
import cv2
import numpy as np
from numpy import ndarray

from transformer.cropping.abstract_cropping_transformer import AbstractCroppingTransformer


class HorizonLevelingTransformer(AbstractCroppingTransformer):
    """Detect and straighten tilted horizons in images."""

    label = "CROP_HORIZON_LEVEL"
    description = "Detect and straighten tilted horizons"

    def transform(self, image: ndarray) -> ndarray:
        """
        Detect horizon line and rotate image to level it.

        Strategy:
        1. Detect edges in the image
        2. Use Hough Line Transform to find strong horizontal lines
        3. Calculate average tilt angle
        4. Rotate image to level horizon
        5. Crop to remove black corners from rotation
        """
        height, width = image.shape[:2]

        # Skip if image too small (need room for rotation crop)
        if height < 600 or width < 600:
            return image

        # Convert to grayscale for edge detection
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # Apply Gaussian blur to reduce noise
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)

        # Edge detection with Canny
        edges = cv2.Canny(blurred, 50, 150, apertureSize=3)

        # Hough Line Transform to detect lines
        # Focus on middle third of image where horizons typically are
        roi_start = height // 3
        roi_end = 2 * height // 3
        roi_edges = edges[roi_start:roi_end, :]

        lines = cv2.HoughLines(roi_edges, 1, np.pi / 180, threshold=100)

        if lines is None:
            return image  # No lines detected, return original

        # Filter for near-horizontal lines (within 10 degrees of horizontal)
        horizontal_angles = []
        for line in lines:
            rho, theta = line[0]
            # Convert theta to degrees, normalize to [-90, 90]
            angle_deg = np.degrees(theta) - 90

            # Keep only near-horizontal lines (-10 to +10 degrees)
            if abs(angle_deg) < 10:
                horizontal_angles.append(angle_deg)

        if len(horizontal_angles) == 0:
            return image  # No horizontal lines found

        # Calculate median angle (more robust than mean)
        tilt_angle = np.median(horizontal_angles)

        # Only correct if tilt is noticeable (> 0.5 degrees)
        if abs(tilt_angle) < 0.5:
            return image

        # Rotate image to level horizon
        center = (width // 2, height // 2)
        rotation_matrix = cv2.getRotationMatrix2D(center, tilt_angle, 1.0)

        # Calculate new bounding box after rotation
        cos = np.abs(rotation_matrix[0, 0])
        sin = np.abs(rotation_matrix[0, 1])
        new_width = int((height * sin) + (width * cos))
        new_height = int((height * cos) + (width * sin))

        # Adjust rotation matrix for new size
        rotation_matrix[0, 2] += (new_width / 2) - center[0]
        rotation_matrix[1, 2] += (new_height / 2) - center[1]

        # Perform rotation
        rotated = cv2.warpAffine(image, rotation_matrix, (new_width, new_height),
                                  flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT,
                                  borderValue=(0, 0, 0))

        # Crop to remove black borders
        # Calculate the largest centered rectangle that fits without black borders
        angle_rad = np.radians(abs(tilt_angle))
        crop_width = int(width * np.cos(angle_rad) + height * np.sin(angle_rad))
        crop_height = int(width * np.sin(angle_rad) + height * np.cos(angle_rad))

        # Use original dimensions as maximum
        crop_width = min(crop_width, width)
        crop_height = min(crop_height, height)

        # Center crop
        start_x = (new_width - crop_width) // 2
        start_y = (new_height - crop_height) // 2

        cropped = rotated[start_y:start_y + crop_height, start_x:start_x + crop_width]

        # Only return result if final size is still reasonable (at least 85% of original)
        if cropped.shape[0] >= height * 0.85 and cropped.shape[1] >= width * 0.85:
            return cropped
        else:
            return image  # Rotation would crop too much, return original
