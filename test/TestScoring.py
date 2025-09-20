import os
from pathlib import Path
from typing import Dict

import cv2
from numpy import ndarray

from juror import Juror
import transformation_agent
from utils.ConfigLoader import ConfigLoader
from utils.Registries import TRANSFORMER_REGISTRY, AGENT_FACTORY_REGISTRY
from utils.TestingUtils import TestingUtils


def main():
    config: Dict = ConfigLoader().load(env=os.environ["ENV_NAME"])
    image_path = Path.cwd() / Path(config['dev']['single_image'])
    # Load image in BGR pixel order
    image: ndarray = TestingUtils.load_image_from_path(image_path)
    juror = Juror()
    score = juror.inference(image_rgb=cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    print(f"Score for image {image_path}: {score}")

    for factory_name in AGENT_FACTORY_REGISTRY.keys():
        for agent in AGENT_FACTORY_REGISTRY.get(factory_name).create_agents():
            image_clone = image.copy()
            transformed_image, label = agent.transform(image_clone)
            score = juror.inference(image_rgb=cv2.cvtColor(transformed_image, cv2.COLOR_BGR2RGB))
            print(f"Score for image with transformation '{label}': {score}")

if __name__ == '__main__':
    main()