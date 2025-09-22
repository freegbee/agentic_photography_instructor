import os
from pathlib import Path
from typing import Dict

from numpy import ndarray

from utils.ConfigLoader import ConfigLoader
from utils.Registries import AGENT_FACTORY_REGISTRY
from utils.TestingUtils import TestingUtils
import transformation_agent


def main():
    config: Dict = ConfigLoader().load(env=os.environ["ENV_NAME"])
    image_path = Path.cwd() / Path(config['dev']['single_image'])
    image: ndarray = TestingUtils.load_image_from_path(image_path)

    for factory_name in AGENT_FACTORY_REGISTRY.keys():
        for agent in AGENT_FACTORY_REGISTRY.get(factory_name).create_agents():
            image_clone = image.copy()
            transformed_image, label = agent.transform(image_clone)
            TestingUtils.save_image_to_path(transformed_image, Path.cwd() / Path(
                config['dev']['temp_outout_dir']) / f"{label}_{image_path.name}")


if __name__ == '__main__':
    main()
