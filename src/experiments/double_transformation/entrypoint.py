import logging
from pathlib import Path

from experiments.double_transformation.DoubleTransformationExperiment import DoubleTransformationExperiment
from utils.LoggingUtils import configure_logging

configure_logging()
logger = logging.getLogger(__name__)


def entrypoint():
    try:
        experiment_name = input("Experiment Name (leer = Double Transformation PoC): ").strip() or "Double Transformation PoC"
    except EOFError:
        experiment_name = "Double Transformation PoC"

    try:
        source_dataset_id = input("Source dataset ID (leer = single_image): ").strip() or "single_image"
    except EOFError:
        source_dataset_id = "single_image"

    try:
        target_directory_name = input("Target Directory Name (leer = double_transformed/<source>): ").strip() or str(Path("double_transformed") / source_dataset_id)
    except EOFError:
        target_directory_name = str(Path("double_transformed") / source_dataset_id)

    try:
        max_images_input = input("Max images (leer = none): ").strip()
    except EOFError:
        max_images_input = ""
    try:
        max_images = int(max_images_input) if max_images_input else None
    except ValueError:
        max_images = None

    try:
        seed_input = input("Seed [42]: ").strip()
    except EOFError:
        seed_input = ""
    try:
        seed = int(seed_input) if seed_input else 42
    except ValueError:
        seed = 42

    try:
        run_name = input("Run name (leer = none): ").strip() or None
    except EOFError:
        run_name = None

    # Transformer sampling options
    try:
        transformer_sample_input = input("Transformer sample size (leer = none): ").strip()
    except EOFError:
        transformer_sample_input = ""
    try:
        transformer_sample_size = int(transformer_sample_input) if transformer_sample_input else None
    except ValueError:
        transformer_sample_size = None

    try:
        transformer_seed_input = input("Transformer sample seed (leer = none): ").strip()
    except EOFError:
        transformer_seed_input = ""
    try:
        transformer_sample_seed = int(transformer_seed_input) if transformer_seed_input else None
    except ValueError:
        transformer_sample_seed = None

    # Batch size for DataLoader / performance tuning
    try:
        batch_size_input = input("Batch size [32]: ").strip()
    except EOFError:
        batch_size_input = ""
    try:
        batch_size = int(batch_size_input) if batch_size_input else 32
    except ValueError:
        batch_size = 32

    # Number of workers for DataLoader (num_workers)
    try:
        num_workers_input = input("Num workers [4]: ").strip()
    except EOFError:
        num_workers_input = ""
    try:
        num_workers = int(num_workers_input) if num_workers_input else 4
    except ValueError:
        num_workers = 4

    logger.info("Starting DoubleTransformationExperiment with experiment_name=%s, source_dataset_id=%s, target_directory=%s, max_images=%s, seed=%s, transformer_sample_size=%s, transformer_sample_seed=%s, batch_size=%s, num_workers=%s", experiment_name, source_dataset_id, target_directory_name, max_images, seed, transformer_sample_size, transformer_sample_seed, batch_size, num_workers)
    exp = DoubleTransformationExperiment(experiment_name=experiment_name, target_directory_root=target_directory_name, run_name=run_name, source_dataset_id=source_dataset_id, max_images=max_images, seed=seed, transformer_sample_size=transformer_sample_size, transformer_sample_seed=transformer_sample_seed, batch_size=batch_size, num_workers=num_workers)
    exp.run()


if __name__ == "__main__":
    from utils import SslHelper

    SslHelper.create_unverified_ssl_context()

    entrypoint()
