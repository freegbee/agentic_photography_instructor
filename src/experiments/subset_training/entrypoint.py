import logging
import os
from pathlib import Path
from typing import List

from dataset.Utils import Utils
from experiments.subset_training.SubsetExperiment import SubsetTraining
from experiments.subset_training.TransformationActor import TransformationActor
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
        topk_input = input("tok images of dataset(leer = 500): ").strip() or None
    except EOFError:
        topk_input = None
    if topk_input is None:
        topk_input = 500
    try:
        topk = int(topk_input)
    except ValueError:
        topk = 500

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

    logger.info("Starting experiment run with experiment_name=%s, run_name=%s, dataset_root=%s, topk %s, batch_size=%d",
                experiment_name, run_name, dataset_root, topk, batch_size)
    absolute_path = Path(os.environ["IMAGE_VOLUME_PATH"]) / dataset_root
    experiment = SubsetTraining(experiment_name, run_name, absolute_path, topk, batch_size)

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
