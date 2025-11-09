import logging

from experiments.shared.logging_config import configure_logging

logger = logging.getLogger(__name__)

def entrypoint():
    configure_logging()

    # Interaktive Eingaben
    try:
        experiment_name = input("Experiment Name (leer = PoC Image Scoring 0.1): ").strip() or None
    except EOFError:
        experiment_name = "PoC Image Scoring 0.1"
    if experiment_name is None:
        experiment_name = "PoC Image Scoring 0.1"

    try:
        dataset_id = input("Dataset ID (leer = single_image): ").strip() or None
    except EOFError:
        dataset_id = "single_image"
    if dataset_id is None:
        dataset_id = "single_image"

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

    logger.info("Starting experiment run with experiment_name=%s, run_name=%s, dataset_id=%s, batch_size=%d", experiment_name, run_name, dataset_id, batch_size)

    from experiments.image_scoring.ImageScoringExperiment import ImageScoringPhotographyExperiment
    exp = ImageScoringPhotographyExperiment(experiment_name=experiment_name, run_name=run_name, dataset_id=dataset_id, batch_size=batch_size)
    exp.run()


if __name__ == "__main__":
    from utils import SslHelper

    SslHelper.create_unverified_ssl_context()

    entrypoint()
