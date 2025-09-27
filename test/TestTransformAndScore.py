import os
from typing import Dict

from torch.utils.data import SequentialSampler, Subset

from data.TransformAndScore import TransformationConfig, TransformAndScore
from utils.ConfigLoader import ConfigLoader
from utils.Registries import init_registries, AGENT_FACTORY_REGISTRY


def main():
    init_registries()
    config: Dict = ConfigLoader().load(env=os.environ["ENV_NAME"])
    agent_factories = AGENT_FACTORY_REGISTRY
    source_dir = config["dev"]["cloned_image_dir"]
    target_dir = config["dev"]["temp_output_dir"]
    # FIXME: Provide Sampler to config. Somehow. Or separate dataset from config and from TransformAndScore
    subsetSampler = None

    transformation_config = TransformationConfig(agent_factories.keys(), source_dir, target_dir, subsetSampler, 2)
    taf = TransformAndScore(transformation_config)
    taf.transform()

if __name__ == "__main__":
    main()