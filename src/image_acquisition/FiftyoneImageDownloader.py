import os
from abc import ABC
from pathlib import Path
from typing import Dict

import fiftyone as fo
import fiftyone.zoo as foz

from utils.ConfigLoader import ConfigLoader


class FiftyoneImageDownloader(ABC):
    def __init__(self, dataset_name):
        if dataset_name is None:
            raise ValueError("dataset_name must be provided")

        self.dataset_name = dataset_name
        config: Dict = ConfigLoader().load(env=os.environ["ENV_NAME"])
        destination_dir = config['data']['fiftyone']['download_dir']
        if destination_dir is None:
            raise ValueError("Set valid directory for fiftyone image downloads in the config file.")
        self.download_dir = Path(destination_dir)

    def download_images(self, split="validation"):
        if not os.path.exists(self.download_dir):
            os.makedirs(self.download_dir)
        fo.config.dataset_zoo_dir = Path(self.download_dir)

        print(f"Downloading dataset images {self.dataset_name} to {self.download_dir} for split {split}")
        dataset = foz.download_zoo_dataset(self.dataset_name, split=split)
        print(f"Downloaded {len(dataset)} images to {self.download_dir}")