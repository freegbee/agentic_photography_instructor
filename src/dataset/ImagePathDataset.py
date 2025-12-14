import os
from pathlib import Path
from typing import List, Tuple

from torch.utils.data import Dataset


class ImagePathDataset(Dataset):
    def __init__(self, images_root_path: Path):
        self.images_root_path: Path = images_root_path
        self.image_files: List[Path] = []
        for root, dirs, files in os.walk(self.images_root_path):
            for file in files:
                if file.endswith('.jpg') or file.endswith('.png'):
                    self.image_files.append(Path(root) / file)

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx) -> Tuple[Path, Path, str]:
        image_path: Path = self.image_files[idx]
        return image_path, image_path.parent, str(image_path.name)