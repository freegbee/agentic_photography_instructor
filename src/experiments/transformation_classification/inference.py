"""
Inference script for transformation classification.
Can predict which transformation(s) were applied to an image using a trained model.
"""
import logging
from pathlib import Path
from typing import List, Tuple

import cv2
import numpy as np
import torch
import torch.nn as nn
from torchvision import models, transforms

logger = logging.getLogger(__name__)


class TransformationClassifier:
    """
    Classifier for predicting which transformation was applied to an image.
    """

    def __init__(self, model_path: Path, device: str = None):
        """
        Args:
            model_path: Path to saved model checkpoint (.pth file)
            device: Device to use (cuda/cpu/mps). If None, auto-detect.
        """
        self.model_path = Path(model_path)

        # Auto-detect device
        if device is None:
            if torch.cuda.is_available():
                self.device = torch.device("cuda")
            elif torch.backends.mps.is_available():
                self.device = torch.device("mps")
            else:
                self.device = torch.device("cpu")
        else:
            self.device = torch.device(device)

        logger.info(f"Using device: {self.device}")

        # Load checkpoint
        checkpoint = torch.load(model_path, map_location=self.device)
        self.transformer_keys = checkpoint['transformer_keys']
        self.num_classes = checkpoint['num_classes']
        self.image_size = checkpoint['image_size']

        logger.info(f"Loaded model with {self.num_classes} classes: {self.transformer_keys}")

        # Create model
        self.model = self._create_model(self.num_classes)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model = self.model.to(self.device)
        self.model.eval()

        # Preprocessing transforms
        self.normalize = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                               std=[0.229, 0.224, 0.225])
        ])

        logger.info("Model loaded and ready for inference")

    def _create_model(self, num_classes: int) -> nn.Module:
        """Create ResNet18 model architecture."""
        model = models.resnet18(pretrained=False)
        num_features = model.fc.in_features
        model.fc = nn.Linear(num_features, num_classes)
        return model

    def predict(self, image_path: Path) -> Tuple[str, float, List[Tuple[str, float]]]:
        """
        Predict transformation for a single image.

        Args:
            image_path: Path to image file

        Returns:
            Tuple of (predicted_transformer, confidence, all_predictions)
            where all_predictions is [(transformer_key, probability), ...]
        """
        # Load and preprocess image
        image = cv2.imread(str(image_path))
        if image is None:
            raise ValueError(f"Failed to load image: {image_path}")

        # Resize
        image = cv2.resize(image, self.image_size)

        # Convert BGR to RGB
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Normalize and convert to tensor
        image_tensor = self.normalize(image_rgb)
        image_tensor = image_tensor.unsqueeze(0).to(self.device)

        # Inference
        with torch.no_grad():
            outputs = self.model(image_tensor)
            probabilities = torch.nn.functional.softmax(outputs, dim=1)
            confidence, predicted_idx = probabilities.max(1)

        predicted_idx = predicted_idx.item()
        confidence = confidence.item()
        predicted_transformer = self.transformer_keys[predicted_idx]

        # Get all predictions sorted by confidence
        all_probs = probabilities[0].cpu().numpy()
        all_predictions = [
            (self.transformer_keys[i], float(all_probs[i]))
            for i in range(len(self.transformer_keys))
        ]
        all_predictions = sorted(all_predictions, key=lambda x: x[1], reverse=True)

        return predicted_transformer, confidence, all_predictions

    def predict_batch(self, image_paths: List[Path]) -> List[Tuple[str, float]]:
        """
        Predict transformations for multiple images.

        Args:
            image_paths: List of image paths

        Returns:
            List of (predicted_transformer, confidence) tuples
        """
        results = []
        for img_path in image_paths:
            predicted, confidence, _ = self.predict(img_path)
            results.append((predicted, confidence))
        return results

    def get_class_names(self) -> List[str]:
        """Get list of transformation class names."""
        return self.transformer_keys


def main():
    """Interactive inference CLI."""
    from utils.LoggingUtils import configure_logging
    configure_logging()

    print("=" * 70)
    print("Transformation Classification - Inference")
    print("=" * 70)

    # Get model path
    try:
        model_path_str = input("\nModel checkpoint path (default: models/transformation_classification/resnet_transformers_2_classes.pth): ").strip()
    except EOFError:
        model_path_str = ""

    if not model_path_str:
        model_path_str = "models/transformation_classification/resnet_transformers_2_classes.pth"

    model_path = Path(model_path_str)
    if not model_path.exists():
        print(f"Error: Model file not found at {model_path}")
        return

    # Load classifier
    print(f"\nLoading model from {model_path}...")
    classifier = TransformationClassifier(model_path)

    print(f"\nModel loaded successfully!")
    print(f"Classes: {', '.join(classifier.get_class_names())}")

    # Inference loop
    while True:
        print("\n" + "-" * 70)
        try:
            image_path_str = input("Image path (or 'quit' to exit): ").strip()
        except EOFError:
            break

        if image_path_str.lower() in ['quit', 'exit', 'q']:
            break

        if not image_path_str:
            continue

        image_path = Path(image_path_str)
        if not image_path.exists():
            print(f"Error: Image not found at {image_path}")
            continue

        try:
            # Predict
            predicted, confidence, all_predictions = classifier.predict(image_path)

            print(f"\nPrediction for: {image_path.name}")
            print(f"  Predicted transformation: {predicted}")
            print(f"  Confidence: {confidence:.4f} ({confidence * 100:.2f}%)")

            print("\nAll predictions:")
            for transformer_key, prob in all_predictions:
                print(f"  {transformer_key}: {prob:.4f} ({prob * 100:.2f}%)")

        except Exception as e:
            print(f"Error during prediction: {e}")
            logger.exception("Prediction error")

    print("\nGoodbye!")


if __name__ == "__main__":
    main()
