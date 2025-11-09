import os
from pathlib import Path
from typing import Tuple, List

from sympy.codegen.ast import Raise
from torch.utils.data import Dataset

from utils.TestingUtils import TestingUtils


class AbsolutePathImageDataset(Dataset[Path]):
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

    def __getitem__(self, idx) -> Tuple[Path, Path, str]:
        image_file_full_path: Path = self.image_files[idx]
        return image_file_full_path, image_file_full_path.parent, str(image_file_full_path.name)
