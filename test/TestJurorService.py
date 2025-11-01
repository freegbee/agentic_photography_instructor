import os
from pathlib import Path
from typing import Dict

from numpy import ndarray

from data_types.AgenticImage import AgenticImage
from juror_client import JurorClient
from utils.ConfigLoader import ConfigLoader
from utils.TestingUtils import TestingUtils

def main():
   config: Dict = ConfigLoader().load(env=os.environ["ENV_NAME"])
   image_path = Path.cwd() / Path(config['dev']['single_image'])
   # Load image in BGR pixel order
   image: ndarray = TestingUtils.load_image_from_path(image_path)

   agentic_image = AgenticImage()
   agentic_image.update_source_image(image, 'BGR', image_path.name)

   juror_client = JurorClient()

   # text: str =juror_client.send_msg("Hello from TestJurorService")
   # print(f"Response from Juror Service: {text}")

   scored = juror_client.score_image(f"{image_path}")
   print(f"scored response: {scored.score}")


if __name__ == '__main__':
    main()