import os
from pathlib import Path
from typing import Dict

import cv2
from numpy import ndarray

from data_types.AgenticImage import AgenticImage
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

    source_image = AgenticImage()
    source_image.add_image(image, 'BGR', True, image_path.name)

    juror = Juror()
    score = juror.inference(source_image.get_image_data('RGB'))
    source_image.score = score

    print(f"Score for image {image_path}: {score}")

    for factory_name in AGENT_FACTORY_REGISTRY.keys():
        for agent in AGENT_FACTORY_REGISTRY.get(factory_name).create_agents():
            changed_image = source_image.clone()
            transformed_image, label = agent.transform(changed_image.get_image_data('BGR'))
            changed_image.add_image(transformed_image, 'BGR', False, image_path)
            changed_image.set_applied_transformers(label)
            score = juror.inference(changed_image.get_image_data('RGB'))
            changed_image.score = score
            print(f"Score for image with transformation '{'->'.join(changed_image.applied_transformers)}': {score}")


if __name__ == '__main__':
    main()
