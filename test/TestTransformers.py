import configparser
import os
from pathlib import Path
from typing import Dict

from numpy import ndarray

from transformer.color_adjustment.GrayscaleTransformer import GrayscaleTransformer
from transformer.cropping.center_square_crop_transformer import CenterSquareCropTransformer
from utils.ConfigLoader import ConfigLoader
from utils.TestingUtils import TestingUtils


def main():
    config: Dict = ConfigLoader().load(env=os.environ["ENV_NAME"])
    image_path = Path.cwd() / Path(config['dev']['single_image'])
    image: ndarray = TestingUtils.load_image_from_path(image_path)
    transformer = GrayscaleTransformer()
    transformer = CenterSquareCropTransformer()
    transformed_image = transformer.transform(image)
    TestingUtils.save_image_to_path(transformed_image, Path.cwd() / Path(config['dev']['temp_output_dir']) / f"{transformer.label}_{image_path.name}")

if __name__ == "__main__":
    main()