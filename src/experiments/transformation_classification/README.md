# Transformation Classification Experiment

Train a ResNet model to classify which image transformation was applied to an image.

## Overview

This experiment enables training a deep learning classifier to identify which transformation(s) were applied to images. It's useful for:
- Understanding transformation effects on images
- Building transformation-aware systems
- Creating automatic image processing pipelines
- Validating transformation detection capabilities

## How It Works

1. **Select Transformations**: Choose 2 or more transformers (e.g., CenterCrop, Grayscale)
2. **Download Dataset**: Automatically downloads the selected image dataset if not already present
3. **Apply Transformations**: Each image gets each transformation applied, creating labeled samples
4. **Train Model**: ResNet18 is trained to classify which transformation was used
5. **Track with MLflow**: All metrics, parameters, and models are logged to MLflow
6. **Inference**: Use trained model to predict transformations on new images

## Usage

### Training

#### In Docker:
```bash
# Start experiment service
cd docker && docker-compose up -d experiment-service

# Run experiment
docker exec -it experiment-service bash
python -m experiments.transformation_classification.entrypoint
```

#### Locally:
```bash
# Set PYTHONPATH
export PYTHONPATH=/path/to/repo/src

# Run experiment
python -m experiments.transformation_classification.entrypoint
```

### Interactive Prompts

The experiment will ask for:
- **Experiment Name**: MLflow experiment name (default: "Transformation Classification")
- **Dataset ID**: Which dataset to use (default: "flickr8k")
- **Run Name**: Optional run name for organization
- **Transformer Selection**: Select 2+ transformers from the available list
- **Batch Size**: Training batch size (default: 32)
- **Epochs**: Number of training epochs (default: 10)
- **Learning Rate**: Optimizer learning rate (default: 0.001)

### Example Session

```
======================================================================
Transformation Classification Experiment
======================================================================

Experiment Name (default: Transformation Classification): My Test
Dataset ID (default: flickr8k): flickr8k
Run name (optional, press Enter to skip):

Available transformers:
  1. C1-1_Center - Creates a squared image centered on the image center.
  2. CA2-1_Grayscale - Converts image to grayscale using luminosity method.
  3. CA3-1_InvertChannel - Inverts the color channels.
  ...

Select at least 2 transformers (comma-separated numbers, e.g., '1,3,5'):
Selection: 1,2

Batch size (default: 32): 32
Number of epochs (default: 10): 5
Learning rate (default: 0.001): 0.001

======================================================================
Configuration Summary:
  Experiment: My Test
  Run: auto-generated
  Dataset: flickr8k
  Transformers: C1-1_Center, CA2-1_Grayscale
  Batch size: 32
  Epochs: 5
  Learning rate: 0.001
======================================================================

Start experiment? (Y/n): y
```

## Inference

After training, use the inference script to predict transformations:

```bash
python -m experiments.transformation_classification.inference
```

Or programmatically:

```python
from pathlib import Path
from experiments.transformation_classification.inference import TransformationClassifier

# Load trained model
classifier = TransformationClassifier(
    model_path=Path("models/transformation_classification/resnet_transformers_2_classes.pth")
)

# Predict single image
predicted, confidence, all_predictions = classifier.predict(Path("test_image.jpg"))
print(f"Predicted: {predicted} (confidence: {confidence:.2%})")

# Predict batch
results = classifier.predict_batch([Path("img1.jpg"), Path("img2.jpg")])
```

## MLflow Tracking

All experiment data is logged to MLflow:

### Parameters Logged:
- `dataset_id`: Source image dataset
- `transformer_keys`: List of transformers used
- `num_transformers`: Number of transformation classes
- `batch_size`, `num_epochs`, `learning_rate`: Training hyperparameters
- `split_ratios`: Train/val/test split ratios
- `device`: Training device (cuda/cpu/mps)
- `total_samples`, `train_samples`, `val_samples`, `test_samples`: Dataset sizes
- `num_classes`: Number of output classes
- `class_names`: Transformation class names
- `total_parameters`, `trainable_parameters`: Model size

### Metrics Logged:
- `train_loss`, `train_accuracy`: Per-epoch training metrics
- `val_loss`, `val_accuracy`: Per-epoch validation metrics
- `test_loss`, `test_accuracy`: Final test set results
- `best_val_accuracy`: Best validation accuracy achieved
- `dataset_download_duration_seconds`: Time to download/verify dataset
- `dataset_creation_duration_seconds`: Time to create transformation dataset
- `epoch_duration_seconds`: Time per epoch
- `total_training_duration_seconds`: Total training time

### Artifacts Logged:
- Model checkpoint (`.pth` file) containing:
  - Model state dict
  - Transformer keys
  - Number of classes
  - Image size

## Architecture

### Components

- **`TransformationClassificationDataset`** (`TransformationClassificationDataset.py`)
  - Loads images from dataset
  - Applies transformations dynamically
  - Creates labeled samples (image, transformation_class)
  - Handles preprocessing and normalization

- **`TransformationClassificationExperiment`** (`TransformationClassificationExperiment.py`)
  - Main experiment orchestrator
  - Downloads dataset if needed
  - Creates and splits dataset
  - Trains ResNet18 model
  - Logs to MLflow
  - Saves model checkpoint

- **`TransformationClassifier`** (`inference.py`)
  - Loads trained model
  - Runs inference on new images
  - Returns predictions with confidence scores

### Model

Uses **ResNet18** (pretrained on ImageNet):
- Final layer replaced with custom classifier
- Input: 224×224 RGB images
- Output: Softmax probabilities over transformation classes
- Optimizer: Adam
- Loss: CrossEntropyLoss

## Configuration

### Dataset Selection

Available datasets are defined in `configs/default.yaml`. Common options:
- `flickr8k`: Flickr 8K dataset
- `div2k`: DIV2K high-quality images
- `places365_validation`: Places365 validation set

Add new datasets by configuring in YAML and implementing download handlers.

### Transformer Selection

All transformers in the registry can be used. Examples:
- **Cropping**: `C1-1_Center`, `C1-2_TopLeft`, etc.
- **Color**: `CA2-1_Grayscale`, `CA3-1_InvertChannel`, `CA4-1_SwapChannel`
- **Image Adjustments**: Various brightness, contrast, etc.

List available transformers:
```python
from utils.Registries import TRANSFORMER_REGISTRY, init_registries
init_registries()
print(list(TRANSFORMER_REGISTRY.keys()))
```

## Extending

### Multi-Transformation Classification

The current implementation trains on single transformations. To extend to **sequences** of transformations:

1. **Modify Dataset**: Apply multiple transformations in sequence:
   ```python
   for t1_idx, t1_key in enumerate(transformer_keys):
       for t2_idx, t2_key in enumerate(transformer_keys):
           # Apply t1 then t2
           # Label as (t1_idx, t2_idx) or combined class
   ```

2. **Adjust Model**:
   - For sequence prediction: Multi-output model or sequence-to-sequence
   - For combined class: Single output with `num_classes = n_transformers ^ sequence_length`

3. **Update Training**: Handle multi-label or multi-output loss functions

### Custom Models

Replace ResNet with other architectures in `_create_resnet_model()`:
```python
from torchvision import models

# Use ResNet50
model = models.resnet50(pretrained=True)

# Use EfficientNet
model = models.efficientnet_b0(pretrained=True)

# Use Vision Transformer
model = models.vit_b_16(pretrained=True)
```

## Requirements

See `requirements.txt` for dependencies. Key requirements:
- PyTorch (with CUDA/MPS support)
- torchvision
- OpenCV (cv2)
- MLflow
- numpy

## Troubleshooting

### Out of Memory
- Reduce `batch_size`
- Use smaller model (ResNet18 is smallest ResNet)
- Reduce `image_size`
- Reduce `num_workers` in DataLoader

### No Images Found
- Verify dataset is downloaded: Check `IMAGE_ACQUISITION_SERVICE_URL` env var
- Check dataset path configuration in `configs/default.yaml`
- Run image acquisition service: `cd docker && docker-compose up -d image-acquisition-service`

### Transformer Not Found
- Ensure transformer is registered: Check `src/transformer/__init__.py`
- Call `init_registries()` before accessing transformers
- Verify transformer key matches `label` class attribute

### Low Accuracy
- Increase `num_epochs` (try 20-50)
- Adjust `learning_rate` (try 0.0001 or 0.01)
- Use more training data
- Ensure transformations are visually distinct
- Check class balance in dataset

## Performance Tips

- **GPU**: Training is much faster on GPU (CUDA or MPS)
- **Workers**: Increase `num_workers` for faster data loading (but watch memory)
- **Batch Size**: Larger batches train faster but need more memory
- **Pretrained**: Using pretrained ResNet significantly improves convergence

## License

Part of the Agentic Photography Instructor project.
