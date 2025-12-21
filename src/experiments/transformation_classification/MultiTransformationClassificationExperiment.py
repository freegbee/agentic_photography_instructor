import logging
import os
import time
from pathlib import Path
from typing import List, Optional, Dict

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from torchvision import models

from data_types.ImageDatasetConfiguration import ImageDatasetConfiguration
from experiments.shared.PhotographyExperiment import PhotographyExperiment
from experiments.shared.Utils import Utils
from experiments.transformation_classification.MultiTransformationClassificationDataset import \
    MultiTransformationClassificationDataset
from utils.ConfigLoader import ConfigLoader
from utils.Registries import init_registries

logger = logging.getLogger(__name__)


class MultiTransformationClassificationExperiment(PhotographyExperiment):
    """
    Experiment to train a ResNet model to classify which transformations were applied to an image.
    
    Uses multi-label classification where the model predicts which N transformations were applied
    in sequence. This avoids the reversible transformation problem by ensuring no transformation
    is applied twice.

    Training flow:
    1. Select transformers and dataset
    2. Download dataset if needed
    3. Apply N unique transformations in sequence to each image
    4. Split into train/val/test sets
    5. Train ResNet model with BCE loss (multi-label)
    6. Log metrics and model to MLflow
    """

    def __init__(self,
                 experiment_name: str,
                 dataset_id: str,
                 transformer_keys: List[str],
                 run_name: Optional[str] = None,
                 batch_size: int = 32,
                 num_epochs: int = 10,
                 learning_rate: float = 0.001,
                 split_ratios: List[float] = None,
                 image_size: tuple = (224, 224),
                 num_workers: int = 4,
                 num_transformations_per_image: int = 2,
                 device: str = None):
        """
        Args:
            experiment_name: MLflow experiment name
            dataset_id: ID of image dataset to use
            transformer_keys: List of transformer labels to use
            run_name: Optional name for this run
            batch_size: Training batch size
            num_epochs: Number of training epochs
            learning_rate: Optimizer learning rate
            split_ratios: [train, val, test] ratios (default: [0.7, 0.15, 0.15])
            image_size: Target image size (width, height)
            num_workers: DataLoader workers
            num_transformations_per_image: Number of unique transformations to apply per image
            device: Device to use (cuda/cpu/mps). If None, auto-detect.
        """
        super().__init__(experiment_name)
        self.dataset_id = dataset_id
        self.transformer_keys = transformer_keys
        self.run_name = run_name
        self.batch_size = batch_size
        self.num_epochs = num_epochs
        self.learning_rate = learning_rate
        self.split_ratios = split_ratios or [0.7, 0.15, 0.15]
        self.image_size = image_size
        self.num_workers = num_workers
        self.num_transformations_per_image = num_transformations_per_image

        # Initialize transformer registry
        init_registries()

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

        # Will be set during run
        self.dataset_config = None
        self.model = None
        self.num_classes = len(transformer_keys)

    def configure(self, config: dict):
        """Optional configuration override."""
        pass

    def _get_tags_for_run(self) -> Dict:
        return {
            "dataset_id": self.dataset_id,
            "transformers": ",".join(self.transformer_keys),
            "experiment_type": "multi_transformation_classification"
        }

    def _get_run_name(self) -> Optional[str]:
        return self.run_name

    def _run_impl(self, experiment_created, active_run):
        """Main experiment logic."""
        logger.info("Starting Multi-Transformation Classification Experiment")

        # Log parameters
        self.log_param("dataset_id", self.dataset_id)
        self.log_param("transformer_keys", self.transformer_keys)
        self.log_param("num_transformers", len(self.transformer_keys))
        self.log_param("batch_size", self.batch_size)
        self.log_param("num_epochs", self.num_epochs)
        self.log_param("learning_rate", self.learning_rate)
        self.log_param("split_ratios", self.split_ratios)
        self.log_param("image_size", self.image_size)
        self.log_param("num_transformations_per_image", self.num_transformations_per_image)
        self.log_param("device", str(self.device))
        self.log_param("classification_type", "multi-label")

        # 1. Ensure dataset is downloaded
        start_time = time.perf_counter()
        config_dict: Dict = ConfigLoader.load_dataset_config(self.dataset_id)
        self.dataset_config = ImageDatasetConfiguration.from_dict(self.dataset_id, config_dict)
        image_dataset_hash = Utils.ensure_image_dataset(self.dataset_config.dataset_id)
        self.log_param("dataset_hash", image_dataset_hash)
        self.log_metric("dataset_download_duration_seconds", time.perf_counter() - start_time)

        images_root_path = self.dataset_config.calculate_images_root_path()
        logger.info(f"Images root path: {images_root_path}")
        self.log_param("images_root_path", str(images_root_path))

        # 2. Create dataset with multi-transformations
        start_time = time.perf_counter()
        full_dataset = MultiTransformationClassificationDataset(
            images_root_path=Path(images_root_path),
            transformer_keys=self.transformer_keys,
            image_size=self.image_size,
            num_transformations_per_image=self.num_transformations_per_image
        )
        self.log_metric("dataset_creation_duration_seconds", time.perf_counter() - start_time)
        self.log_param("total_samples", len(full_dataset))
        self.log_param("num_classes", full_dataset.get_num_classes())
        self.log_param("class_names", full_dataset.get_class_names())

        # 3. Split dataset
        train_size = int(self.split_ratios[0] * len(full_dataset))
        val_size = int(self.split_ratios[1] * len(full_dataset))
        test_size = len(full_dataset) - train_size - val_size

        train_dataset, val_dataset, test_dataset = random_split(
            full_dataset,
            [train_size, val_size, test_size],
            generator=torch.Generator().manual_seed(42)
        )

        logger.info(f"Dataset split: train={train_size}, val={val_size}, test={test_size}")
        self.log_param("train_samples", train_size)
        self.log_param("val_samples", val_size)
        self.log_param("test_samples", test_size)

        # 4. Create data loaders
        train_loader = DataLoader(
            train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=True if self.device.type == "cuda" else False
        )

        val_loader = DataLoader(
            val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True if self.device.type == "cuda" else False
        )

        test_loader = DataLoader(
            test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True if self.device.type == "cuda" else False
        )

        # 5. Create ResNet model with frozen backbone
        self.model = self._create_resnet_model(self.num_classes)
        self.model = self.model.to(self.device)

        # Count parameters
        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        self.log_param("total_parameters", total_params)
        self.log_param("trainable_parameters", trainable_params)
        logger.info(f"Model parameters: {trainable_params:,} trainable / {total_params:,} total")

        # 6. Setup training - use BCEWithLogitsLoss for multi-label classification
        criterion = nn.BCEWithLogitsLoss()
        optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)

        # 7. Train model
        logger.info("Starting training...")
        start_training = time.perf_counter()

        best_val_f1 = 0.0
        for epoch in range(self.num_epochs):
            epoch_start = time.perf_counter()

            # Training phase
            train_loss, train_metrics = self._train_epoch(
                train_loader, criterion, optimizer, epoch
            )

            # Validation phase
            val_loss, val_metrics = self._validate_epoch(
                val_loader, criterion, epoch
            )

            epoch_duration = time.perf_counter() - epoch_start

            # Log metrics
            self.log_metric("train_loss", train_loss, step=epoch)
            self.log_metric("train_precision", train_metrics['precision'], step=epoch)
            self.log_metric("train_recall", train_metrics['recall'], step=epoch)
            self.log_metric("train_f1", train_metrics['f1'], step=epoch)
            self.log_metric("train_exact_match", train_metrics['exact_match'], step=epoch)
            
            self.log_metric("val_loss", val_loss, step=epoch)
            self.log_metric("val_precision", val_metrics['precision'], step=epoch)
            self.log_metric("val_recall", val_metrics['recall'], step=epoch)
            self.log_metric("val_f1", val_metrics['f1'], step=epoch)
            self.log_metric("val_exact_match", val_metrics['exact_match'], step=epoch)
            self.log_metric("epoch_duration_seconds", epoch_duration, step=epoch)

            logger.info(
                f"Epoch {epoch + 1}/{self.num_epochs}: "
                f"train_loss={train_loss:.4f}, train_f1={train_metrics['f1']:.4f}, "
                f"val_loss={val_loss:.4f}, val_f1={val_metrics['f1']:.4f}, "
                f"duration={epoch_duration:.2f}s"
            )

            # Save best model based on validation F1
            if val_metrics['f1'] > best_val_f1:
                best_val_f1 = val_metrics['f1']
                self.log_metric("best_val_f1", best_val_f1)
                logger.info(f"New best validation F1: {best_val_f1:.4f}")

        training_duration = time.perf_counter() - start_training
        self.log_metric("total_training_duration_seconds", training_duration)

        # 8. Test model
        logger.info("Evaluating on test set...")
        test_loss, test_metrics = self._validate_epoch(test_loader, criterion, epoch=-1)
        self.log_metric("test_loss", test_loss)
        self.log_metric("test_precision", test_metrics['precision'])
        self.log_metric("test_recall", test_metrics['recall'])
        self.log_metric("test_f1", test_metrics['f1'])
        self.log_metric("test_exact_match", test_metrics['exact_match'])
        logger.info(
            f"Test results: loss={test_loss:.4f}, "
            f"precision={test_metrics['precision']:.4f}, "
            f"recall={test_metrics['recall']:.4f}, "
            f"f1={test_metrics['f1']:.4f}, "
            f"exact_match={test_metrics['exact_match']:.4f}"
        )

        # 9. Save model
        model_path = self._save_model()
        self.log_artifact(str(model_path))
        logger.info(f"Model saved to {model_path}")

        logger.info("Multi-Transformation Classification Experiment completed successfully!")

    def _create_resnet_model(self, num_classes: int) -> nn.Module:
        """
        Create ResNet18 model with frozen backbone and custom multi-label classification head.
        
        Uses sigmoid activation for multi-label classification (multiple transformations can be active).
        """
        # Load pretrained ResNet18
        try:
            weights = models.ResNet18_Weights.DEFAULT
            model = models.resnet18(weights=weights)
        except Exception:
            model = models.resnet18(pretrained=True)
        
        # Get feature dimension and replace fc with identity
        feature_dim = int(model.fc.in_features)  # 512 for ResNet18
        model.fc = nn.Identity()
        
        # Freeze all backbone parameters
        for param in model.parameters():
            param.requires_grad = False
        
        # Create a wrapper that combines backbone + multi-label classification head
        class FrozenBackboneMultiLabelClassifier(nn.Module):
            def __init__(self, backbone, feature_dim, num_classes):
                super(FrozenBackboneMultiLabelClassifier, self).__init__()
                self.backbone = backbone

                # Multi-label classification head (no softmax, use sigmoid in loss)
                # self.head = nn.Sequential(
                #     nn.Linear(feature_dim, 128),
                #     nn.ReLU(),
                #     nn.Linear(128, num_classes)
                #     # No activation here - BCEWithLogitsLoss includes sigmoid
                # )

                # IMPROVED multi-label classification head
                self.head = nn.Sequential(
                    nn.Linear(feature_dim, 512),
                    nn.BatchNorm1d(512),
                    nn.ReLU(),
                    nn.Dropout(0.5),

                    nn.Linear(512, 256),
                    nn.BatchNorm1d(256),
                    nn.ReLU(),
                    nn.Dropout(0.4),

                    nn.Linear(256, 128),
                    nn.BatchNorm1d(128),
                    nn.ReLU(),
                    nn.Dropout(0.3),

                    nn.Linear(128, num_classes)
                    # No activation here - BCEWithLogitsLoss includes sigmoid
                )
            
            def forward(self, x):
                # Backbone is frozen, so no gradients computed here
                with torch.no_grad():
                    features = self.backbone(x)
                # Only head is trainable
                return self.head(features)
        
        classifier = FrozenBackboneMultiLabelClassifier(model, feature_dim, num_classes)
        
        logger.info(f"Created ResNet18 with frozen backbone and trainable multi-label head")
        logger.info(f"Feature dim: {feature_dim}, Output classes: {num_classes}")
        logger.info(f"Head architecture: Linear({feature_dim}, 128) -> ReLU -> Linear(128, {num_classes})")
        
        return classifier

    def _train_epoch(self, dataloader, criterion, optimizer, epoch) -> tuple:
        """Train for one epoch."""
        self.model.train()
        running_loss = 0.0
        
        all_preds = []
        all_labels = []

        for batch_idx, (images, labels) in enumerate(dataloader):
            images = images.to(self.device)
            labels = labels.to(self.device)

            # Forward pass
            optimizer.zero_grad()
            outputs = self.model(images)
            loss = criterion(outputs, labels)

            # Backward pass
            loss.backward()
            optimizer.step()

            # Statistics
            running_loss += loss.item()
            
            # Convert logits to predictions (threshold at 0.5)
            preds = (torch.sigmoid(outputs) > 0.5).float()
            all_preds.append(preds.cpu())
            all_labels.append(labels.cpu())

        avg_loss = running_loss / len(dataloader)
        
        # Calculate metrics
        all_preds = torch.cat(all_preds, dim=0)
        all_labels = torch.cat(all_labels, dim=0)
        metrics = self._calculate_multilabel_metrics(all_preds, all_labels)
        
        return avg_loss, metrics

    def _validate_epoch(self, dataloader, criterion, epoch) -> tuple:
        """Validate/test for one epoch."""
        self.model.eval()
        running_loss = 0.0
        
        all_preds = []
        all_labels = []

        with torch.no_grad():
            for images, labels in dataloader:
                images = images.to(self.device)
                labels = labels.to(self.device)

                outputs = self.model(images)
                loss = criterion(outputs, labels)

                running_loss += loss.item()
                
                # Convert logits to predictions
                preds = (torch.sigmoid(outputs) > 0.5).float()
                all_preds.append(preds.cpu())
                all_labels.append(labels.cpu())

        avg_loss = running_loss / len(dataloader)
        
        # Calculate metrics
        all_preds = torch.cat(all_preds, dim=0)
        all_labels = torch.cat(all_labels, dim=0)
        metrics = self._calculate_multilabel_metrics(all_preds, all_labels)
        
        return avg_loss, metrics

    def _calculate_multilabel_metrics(self, preds: torch.Tensor, labels: torch.Tensor) -> Dict[str, float]:
        """Calculate multi-label classification metrics."""
        # Per-sample metrics
        tp = (preds * labels).sum(dim=1)  # True positives per sample
        fp = (preds * (1 - labels)).sum(dim=1)  # False positives per sample
        fn = ((1 - preds) * labels).sum(dim=1)  # False negatives per sample
        
        # Avoid division by zero
        precision = tp / (tp + fp + 1e-10)
        recall = tp / (tp + fn + 1e-10)
        f1 = 2 * (precision * recall) / (precision + recall + 1e-10)
        
        # Exact match (all labels must match)
        exact_match = (preds == labels).all(dim=1).float()
        
        return {
            'precision': precision.mean().item(),
            'recall': recall.mean().item(),
            'f1': f1.mean().item(),
            'exact_match': exact_match.mean().item()
        }

    def _save_model(self) -> Path:
        """Save model checkpoint."""
        model_dir = Path("models") / "multi_transformation_classification"
        model_dir.mkdir(parents=True, exist_ok=True)

        model_filename = f"resnet_multi_{len(self.transformer_keys)}_classes_{self.num_transformations_per_image}_per_image.pth"
        model_path = model_dir / model_filename

        torch.save({
            'model_state_dict': self.model.state_dict(),
            'transformer_keys': self.transformer_keys,
            'num_classes': self.num_classes,
            'num_transformations_per_image': self.num_transformations_per_image,
            'image_size': self.image_size,
        }, model_path)

        return model_path
