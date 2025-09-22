import os
from pathlib import Path
from typing import Dict

from numpy import ndarray

import transformer.cropping as cropping
from transformer.AbstractTransformer import AbstractTransformer
from utils.ConfigLoader import ConfigLoader
from utils.Registries import TRANSFORMER_REGISTRY
from utils.TestingUtils import TestingUtils

if __name__ == "__main__":
    at = AbstractTransformer
    subclasses = list(TRANSFORMER_REGISTRY.keys())
    print(subclasses)
    transformer = TRANSFORMER_REGISTRY.get(cropping.CenterSquareCropTransformer.label)
    config: Dict = ConfigLoader().load(env=os.environ["ENV_NAME"])
    image_path = Path.cwd() / Path(config['dev']['single_image'])
    image: ndarray = TestingUtils.load_image_from_path(image_path)
    transformed_image = transformer.transform(image)
    TestingUtils.save_image_to_path(transformed_image, Path.cwd() / Path(
        config['dev']['temp_output_dir']) / f"{transformer.label}_{image_path.name}")
