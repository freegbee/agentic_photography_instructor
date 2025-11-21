import logging
import os
from pathlib import Path

from pyarrow.compute import top_k_unstable

from experiments.subset_training.SubsetExperiment import SubsetTraining
from utils.LoggingUtils import configure_logging

configure_logging()
logger = logging.getLogger(__name__)


def entrypoint():
    # Interaktive Eingaben
    try:
        experiment_name = input("Experiment Name (leer = PoC Subset Training 0.1): ").strip() or None
    except EOFError:
        experiment_name = "PoC Subset Training 0.1"
    if experiment_name is None:
        experiment_name = "PoC Subset Training 0.1"

    try:
        dataset_root = input("dataset root folder (leer = div2k_twenty_images): ").strip() or None
    except EOFError:
        dataset_root = None
    if dataset_root is None:
        dataset_root = "flickr8k"

    try:
        top_k_input = input("top k images of dataset(leer = 500): ").strip() or None
    except EOFError:
        top_k_input = None
    if top_k_input is None:
        top_k_input = 500
    try:
        top_k = int(top_k_input)
    except ValueError:
        top_k = 500

    try:
        run_name = input("Run name (leer = none): ").strip() or None
    except EOFError:
        run_name = None

    try:
        batch_input = input("Batch size [20]: ").strip()
    except EOFError:
        batch_input = ""
    try:
        batch_size = int(batch_input) if batch_input else 20
    except ValueError:
        batch_size = 20

    logger.info("Starting experiment run with experiment_name=%s, run_name=%s, dataset_root=%s, top_k %s, batch_size=%d",
                experiment_name, run_name, dataset_root, top_k, batch_size)
    absolute_path = Path(os.environ["IMAGE_VOLUME_PATH"]) / dataset_root
    experiment = SubsetTraining(experiment_name, run_name, absolute_path, top_k, batch_size)

    experiment.run()

    # dl = Utils.create_topk_coco_dataloader(absolute_path, batch_size=batch_size, k=int(20))
    # ta: TransformationActor = TransformationActor()
    # result_csv: List[str] = ["image_id,image_relative_path,score_before,score_after,score_change,transformation"]
    # for batch in dl:
    #     for img_data in batch:
    #         logger.debug(" ImageData: id=%s, path=%s, score=%.4f", img_data.id, img_data.image_relative_path,
    #                     img_data.score)
    #         transformed, score = ta.transform_and_score(img_data.get_image_data(), transformer_name)
    #         logger.info("Transformation for image %s Score before=%.4f , score after=%.4f  --> change=%.4f", img_data.image_relative_path, img_data.score, score, score - img_data.score)
    #         result_csv.append(f"{img_data.id},{img_data.image_relative_path},{img_data.score:.4f},{score:.4f},{score - img_data.score:.4f},{transformer_name}")
    #
    # for line in result_csv:
    #     print(line)

    print("Experiment completed.")


if __name__ == "__main__":
    from utils import SslHelper

    SslHelper.create_unverified_ssl_context()

    entrypoint()
