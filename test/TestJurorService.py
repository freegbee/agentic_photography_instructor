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

   with JurorClient(base_url=os.environ["JUROR_SERVICE_URL"]) as juror_client:
       logger.info(f"Mehrfaches Scoring des Bildes {image_path} umd erstmal einfach etwas mehr calls zu machen...")
       for i in range(4):
            scored = juror_client.score_image(f"{image_path}")
            logger.info(f"{i} scored response: {scored.score}")


if __name__ == '__main__':
    main()