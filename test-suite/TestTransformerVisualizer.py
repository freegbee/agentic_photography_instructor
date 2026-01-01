import math
import os
import cv2
import numpy as np
import pytest

from transformer.AbstractTransformer import TRANSFORMER_REGISTRY


def test_generate_transformer_mosaic():
    # Path to resources
    resource_dir = os.path.join(os.path.dirname(__file__), "resources")
    image_path = os.path.join(resource_dir, "test_landscape.png")
    output_path = os.path.join(resource_dir, "transformer_mosaic.jpg")

    # Load image
    if not os.path.exists(image_path):
        pytest.skip(f"Test image not found at {image_path}")
    
    original_image = cv2.imread(image_path)
    if original_image is None:
        pytest.fail(f"Failed to load image from {image_path}")

    # Resize for the mosaic to keep it manageable (e.g., width 200px)
    tile_width = 200
    aspect_ratio = original_image.shape[0] / original_image.shape[1]
    tile_height = int(tile_width * aspect_ratio)
    
    # Text area height
    text_height = 30
    
    small_image = cv2.resize(original_image, (tile_width, tile_height))

    # Get all transformers
    transformers = [cls() for cls in TRANSFORMER_REGISTRY.values()]
    # Sort by label for consistent order
    transformers.sort(key=lambda x: x.label)

    # Add original image as the first item
    results = [("ORIGINAL", small_image.copy())]

    for t in transformers:
        try:
            # Apply transform
            transformed = t.transform(small_image.copy())
            
            # Ensure result has same dimensions (some might crop)
            # If size changed, resize back to tile size to fit mosaic
            if transformed.shape[0] != tile_height or transformed.shape[1] != tile_width:
                transformed = cv2.resize(transformed, (tile_width, tile_height))
            
            results.append((t.label, transformed))
        except Exception as e:
            print(f"Failed to apply {t.label}: {e}")
            # Create an error tile
            error_img = np.zeros((tile_height, tile_width, 3), dtype=np.uint8)
            cv2.putText(error_img, "ERROR", (10, tile_height//2), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
            results.append((t.label, error_img))

    # Calculate mosaic dimensions
    cols = 5
    rows = math.ceil(len(results) / cols)

    # Create canvas
    mosaic_width = cols * tile_width
    mosaic_height = rows * (tile_height + text_height)
    mosaic = np.ones((mosaic_height, mosaic_width, 3), dtype=np.uint8) * 255  # White background

    for i, (label, img) in enumerate(results):
        r = i // cols
        c = i % cols

        x = c * tile_width
        y = r * (tile_height + text_height)

        # Place image
        mosaic[y:y+tile_height, x:x+tile_width] = img

        # Place label
        # Font settings
        font_scale = 0.4
        thickness = 1
        text_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)[0]
        text_x = x + (tile_width - text_size[0]) // 2
        text_y = y + tile_height + 20
        
        cv2.putText(mosaic, label, (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 0, 0), thickness)

    # Save result
    cv2.imwrite(output_path, mosaic)
    print(f"Mosaic saved to {output_path}")