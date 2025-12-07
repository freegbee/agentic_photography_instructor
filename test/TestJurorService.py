import logging
import os
from pathlib import Path
from typing import Dict

from numpy import ndarray

from data_types.AgenticImage import AgenticImage
from juror_client import JurorClient
from utils.ConfigLoader import ConfigLoader
from utils.TestingUtils import TestingUtils

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


def main():
    config: Dict = ConfigLoader().load(env=os.environ["ENV_NAME"])
    image_path = Path.cwd() / Path(config['dev']['single_image'])
    # Load image in BGR pixel order
    image: ndarray = TestingUtils.load_image_from_path(image_path)

    agentic_image = AgenticImage()
    agentic_image.update_source_image(image, 'BGR', image_path.name)

    with JurorClient(base_url=os.environ["JUROR_SERVICE_URL"], use_cache=False) as juror_client:
        logger.info(f"Mehrfaches Scoring des Bildes {image_path} um erstmal einfach etwas mehr calls zu machen...")
        for i in range(4):
            scored = juror_client.score_image(f"{image_path}")
            logger.info(f"{i} scored response from filebased scoring: {scored.score}")
            scored_ndarray = juror_client.score_ndarray_rgb(agentic_image.source_image.get_image_data("RGB"))
            logger.info(f"{i} scored response from ndarray scoring: {scored_ndarray.score}")
            scored_ndarray_bgr = juror_client.score_ndarray_bgr(image)
            logger.info(f"{i} scored response from ndarray scoring with BGR: {scored_ndarray_bgr.score}")
            if scored != scored_ndarray or scored != scored_ndarray_bgr:
                logger.error("Inconsistent scoring results between different methods!")



if __name__ == '__main__':
    main()
