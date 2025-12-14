"""
Helper script to generate RL training dataset with train/val/test splits.

This script uses the ImageDegradationExperiment with split support to create
degraded images for RL training.
"""

import logging
import os

from experiments.double_transformation import DoubleTransformationExperiment
from utils.LoggingUtils import configure_logging

configure_logging()
logger = logging.getLogger(__name__)


def generate_rl_doubletransformed_dataset():
    """Generate RL training dataset with splits."""

    # Configuration
    try:
        experiment_name = input("Experiment Name [RL DoubleTransform Dataset Generation]: ").strip()
    except EOFError:
        experiment_name = ""
    if not experiment_name:
        experiment_name = "RL DoubleTransform Dataset Generation"

    try:
        source_dataset = input("Source dataset ID [flickr8k]: ").strip()
    except EOFError:
        source_dataset = ""
    if not source_dataset:
        source_dataset = "flickr8k"

    try:
        output_dir = input("Output directory [rl_double_training_data]: ").strip()
    except EOFError:
        output_dir = ""
    if not output_dir:
        output_dir = "rl_double_training_data"

    try:
        use_random = input("Use random transformers? [Y/n]: ").strip().lower()
    except EOFError:
        use_random = "y"
    use_random_transformers = use_random != "n"

    if not use_random_transformers:
        transformer_name = input("Transformer name: ").strip()
    else:
        transformer_name = "RANDOM"

    try:
        train_ratio = float(input("Train ratio [0.7]: ").strip() or "0.7")
        val_ratio = float(input("Validation ratio [0.15]: ").strip() or "0.15")
        test_ratio = float(input("Test ratio [0.15]: ").strip() or "0.15")
    except (EOFError, ValueError):
        train_ratio, val_ratio, test_ratio = 0.7, 0.15, 0.15

    # Validate ratios
    if abs(train_ratio + val_ratio + test_ratio - 1.0) > 0.01:
        logger.warning("Split ratios don't sum to 1.0, normalizing...")
        total = train_ratio + val_ratio + test_ratio
        train_ratio /= total
        val_ratio /= total
        test_ratio /= total

    try:
        batch_size = int(input("Batch size [16]: ").strip() or "16")
    except (EOFError, ValueError):
        batch_size = 16

    try:
        seed = int(input("Random seed [42]: ").strip() or "42")
    except (EOFError, ValueError):
        seed = 42

    logger.info("=" * 80)
    logger.info("RL Dataset Generation Configuration")
    logger.info("=" * 80)
    logger.info(f"Experiment Name: {experiment_name}")
    logger.info(f"Source Dataset: {source_dataset}")
    logger.info(f"Output Directory: {output_dir}")
    logger.info(f"Transformer: {transformer_name}")
    logger.info(f"Split Ratios: train={train_ratio:.2f}, val={val_ratio:.2f}, test={test_ratio:.2f}")
    logger.info(f"Batch Size: {batch_size}")
    logger.info(f"Seed: {seed}")
    logger.info("=" * 80)

    # Create experiment
    experiment = DoubleTransformationExperiment(
        experiment_name=experiment_name,
        target_directory_root=output_dir,
        source_dataset_id=source_dataset,
        batch_size=batch_size,
        split_ratios=(train_ratio, val_ratio, test_ratio),
        seed=seed,
    )

    # Run experiment
    logger.info("Starting dataset generation...")
    experiment.run()

    logger.info("=" * 80)
    logger.info("Dataset generation complete!")
    logger.info("=" * 80)
    logger.info(f"Dataset saved to: {os.environ['IMAGE_VOLUME_PATH']}/{output_dir}")
    logger.info("Directory structure:")
    logger.info(f"  {output_dir}/")
    logger.info(f"    ├── train/")
    logger.info(f"    │   ├── images/")
    logger.info(f"    │   └── annotations.json")
    logger.info(f"    ├── val/")
    logger.info(f"    │   ├── images/")
    logger.info(f"    │   └── annotations.json")
    logger.info(f"    └── test/")
    logger.info(f"        ├── images/")
    logger.info(f"        └── annotations.json")
    logger.info("=" * 80)
    logger.info("You can now train the RL agent using:")
    logger.info(f"  python -m experiments.rl_training.entrypoint")
    logger.info("=" * 80)


if __name__ == "__main__":
    from utils import SslHelper

    SslHelper.create_unverified_ssl_context()

    generate_rl_doubletransformed_dataset()
