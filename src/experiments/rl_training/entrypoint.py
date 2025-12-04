import logging
import os
from pathlib import Path

from experiments.rl_training.RLTrainingExperiment import RLTrainingExperiment
from utils.LoggingUtils import configure_logging

configure_logging()
logger = logging.getLogger(__name__)


def entrypoint():
    """
    Entrypoint for RL training experiment.

    Prompts for configuration and starts training.
    """
    # Experiment configuration
    try:
        experiment_name = input("Experiment Name [RL Image Optimization]: ").strip()
    except EOFError:
        experiment_name = ""
    if not experiment_name:
        experiment_name = "RL Image Optimization"

    try:
        run_name = input("Run name (optional): ").strip() or None
    except EOFError:
        run_name = None

    # Dataset configuration
    try:
        dataset_root_input = input("Dataset root folder [rl_training_data]: ").strip()
    except EOFError:
        dataset_root_input = ""
    if not dataset_root_input:
        dataset_root_input = "rl_training_data"

    dataset_root = Path(os.environ["IMAGE_VOLUME_PATH"]) / dataset_root_input

    # Training hyperparameters
    try:
        max_steps = int(input("Max steps per episode [5]: ").strip() or "5")
    except (EOFError, ValueError):
        max_steps = 5

    try:
        num_epochs = int(input("Number of training epochs [10]: ").strip() or "10")
    except (EOFError, ValueError):
        num_epochs = 10

    try:
        batch_size = int(input("Replay buffer batch size [32]: ").strip() or "32")
    except (EOFError, ValueError):
        batch_size = 32

    try:
        dataloader_batch_size = int(input("DataLoader batch size [8]: ").strip() or "8")
    except (EOFError, ValueError):
        dataloader_batch_size = 8

    try:
        learning_rate = float(input("Learning rate [0.001]: ").strip() or "0.001")
    except (EOFError, ValueError):
        learning_rate = 0.001

    try:
        add_stop = input("Add STOP action? [y/N]: ").strip().lower() == 'y'
    except EOFError:
        add_stop = False

    logger.info("=" * 80)
    logger.info("Starting RL Training Experiment")
    logger.info("=" * 80)
    logger.info(f"Experiment: {experiment_name}")
    logger.info(f"Run: {run_name}")
    logger.info(f"Dataset: {dataset_root}")
    logger.info(f"Max steps per episode: {max_steps}")
    logger.info(f"Num epochs: {num_epochs}")
    logger.info(f"Batch size: {batch_size}")
    logger.info(f"DataLoader batch size: {dataloader_batch_size}")
    logger.info(f"Learning rate: {learning_rate}")
    logger.info(f"Add STOP action: {add_stop}")
    logger.info("=" * 80)

    # Create and run experiment
    experiment = RLTrainingExperiment(
        experiment_name=experiment_name,
        run_name=run_name,
        dataset_root=dataset_root,
        max_steps_per_episode=max_steps,
        batch_size=batch_size,
        num_epochs=num_epochs,
        dataloader_batch_size=dataloader_batch_size,
        learning_rate=learning_rate,
        add_stop_action=add_stop,
    )

    experiment.run()

    logger.info("=" * 80)
    logger.info("Training completed successfully!")
    logger.info("=" * 80)


if __name__ == "__main__":
    from utils import SslHelper

    SslHelper.create_unverified_ssl_context()

    entrypoint()
