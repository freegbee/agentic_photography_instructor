import os
from pathlib import Path

import numpy as np
from sympy.codegen.ast import Raise
from torch.utils.data import Dataset

from utils.TestingUtils import TestingUtils


class BasicTestDataset(Dataset[np.ndarray]):
    """
    A basic data loader for testing purposes. It implements the Dataset interface and uses some random files for testing.

    Implementation is following https://www.codegenes.net/blog/pytorch-load-image-from-folder/
    """

    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        if transform is not None:
            Raise("Transform not supported for this data loader")
        self.transform = transform
        self.image_files: List[Path] = []
        for root, dirs, files in os.walk(root_dir):
            for file in files:
                if file.endswith('.jpg') or file.endswith('.png'):
                    self.image_files.append(Path(root) / file)

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        image_path: Path = self.image_files[idx]
        return TestingUtils.load_image_from_path(image_path), image_path.parent, str(image_path.name)
