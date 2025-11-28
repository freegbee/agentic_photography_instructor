import logging
import uuid
from pathlib import Path

from experiments.image_degrade.ImageDegradationExperiment import ImageDegradationExperiment
from utils.LoggingUtils import configure_logging

configure_logging()
logger = logging.getLogger(__name__)

def entrypoint():
    # Interaktive Eingaben
    try:
        experiment_name = input("Experiment Name (leer = PoC Image Degrading 0.1): ").strip() or None
    except EOFError:
        experiment_name = "PoC Image Degrading 0.6"
    if experiment_name is None:
        experiment_name = "PoC Image Degrading 0.6"

    try:
        source_dataset_id = input("Source dataset ID (leer = single_image): ").strip() or None
    except EOFError:
        source_dataset_id = "single_image"
    if source_dataset_id is None:
        source_dataset_id = "single_image"

    try:
        transformer_name = input("Transformer name (leer = CA_INV_B; RANDOM=Zufälliger, umkehrbarer Transformer): ").strip() or None
    except EOFError:
        transformer_name = "CA_INV_B"
    if transformer_name is None:
        transformer_name = "CA_INV_B"

    try:
        target_directory_name = input("Target Directory Name (leer = bilder root + degraded + source dataset id + transformer): ").strip() or None
    except EOFError:
        target_directory_name = None
    if target_directory_name is None:
        target_directory_name = str(Path("degraded") / source_dataset_id / transformer_name)

    try:
        target_dataset_id = input("Target dataset ID (leer = neue UUID): ").strip() or None
    except EOFError:
        target_dataset_id = uuid.uuid4()
    if target_dataset_id is None:
        target_dataset_id = uuid.uuid4()

    try:
        run_name = input("Run name (leer = none): ").strip() or None
    except EOFError:
        run_name = None

    try:
        batch_input = input("Batch size [4]: ").strip()
    except EOFError:
        batch_input = ""
    try:
        batch_size = int(batch_input) if batch_input else 4
    except ValueError:
        batch_size = 4


    logger.info("Starting experiment run with experiment_name=%s, run_name=%s, target_directory_name=%s, target_dataset_id=%s, transformer=%s, source_dataset_id=%s, batch_size=%d", experiment_name, run_name, target_directory_name, target_dataset_id, transformer_name, source_dataset_id, batch_size)
    exp = ImageDegradationExperiment(experiment_name=experiment_name, target_directory_root=target_directory_name, target_dataset_id=target_dataset_id, transformer_name=transformer_name, run_name=run_name, source_dataset_id=source_dataset_id, batch_size=batch_size)
    exp.run()


if __name__ == "__main__":
    from utils import SslHelper

    SslHelper.create_unverified_ssl_context()

    entrypoint()
