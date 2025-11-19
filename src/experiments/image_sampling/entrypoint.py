import logging
import os
from pathlib import Path

from dataset.Utils import Utils
from utils.LoggingUtils import configure_logging

configure_logging()
logger = logging.getLogger(__name__)

def entrypoint():
    # Interaktive Eingaben
    try:
        dataset_root = input("dataset root folder (leer = degraded/twenty_images/CA_INV_B): ").strip() or None
    except EOFError:
        dataset_root = None
    if dataset_root is None:
        dataset_root = "degraded/twenty_images/CA_INV_B"

    try:
        batch_input = input("Batch size [4]: ").strip()
    except EOFError:
        batch_input = 4
    if batch_input == "":
        batch_input = 4

    try:
        k_input = input("K [leer = 5]: ").strip() or None
    except EOFError:
        k_input = ""
    if k_input == "" or k_input is None:
        k_input = 5

    absolute_path = Path(os.environ["IMAGE_VOLUME_PATH"]) / dataset_root

    logger.info("about to create dataloader with dataset_root=%s, batch_size=%s, k=%s", absolute_path, batch_input, k_input)

    dl = Utils.create_topk_coco_dataloader(absolute_path, batch_size=int(batch_input), k=int(k_input))
    for batch in dl:
        for img_data in batch:
            logger.info(" ImageData: id=%s, path=%s, score=%.4f", img_data.id, img_data.image_relative_path, img_data.score)


if __name__ == "__main__":
    from utils import SslHelper

    SslHelper.create_unverified_ssl_context()

    entrypoint()
