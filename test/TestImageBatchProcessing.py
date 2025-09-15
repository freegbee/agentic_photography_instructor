import os
from pathlib import Path
from typing import Dict

from torch.utils.data import DataLoader, Dataset, RandomSampler, SequentialSampler, Subset

from image_aquisition.BasicTestDataset import BasicTestDataset
from utils.ConfigLoader import ConfigLoader
from utils.Registries import AGENT_FACTORY_REGISTRY
from transformation_agent import StaticTransformationAgentFactory
from utils.TestingUtils import TestingUtils


def collate_keep_size(batch):
    # Returns a batch without stacking the images, so that they can keep their original size
    return batch


def main():
    config: Dict = ConfigLoader().load(env=os.environ["ENV_NAME"])
    image_dir = config["dev"]["cloned_image_dir"]
    dataset: Dataset = BasicTestDataset(image_dir)
    print(f"Number of images in {image_dir}: {len(dataset)}")

    # sampler to return only 10 random samples from the dataset. Not sure what replacement=True means
    num_samples = min(10, len(dataset))
    rndSampler = RandomSampler(dataset, replacement=True, num_samples=num_samples)
    subsetSampler = SequentialSampler(Subset(dataset, range(num_samples)))

    # collate_keep_size sorgt dafür, dass die Bilder ihre Originalgrösse behalten können. Standardmässig müssten alle Bilder die gleiche Grösse haben, was wir aber nicht wollen, da wir hin diesem Moment die Bilder untransformiert lassen wollen.
    dataloader = DataLoader(dataset, batch_size=2, collate_fn=collate_keep_size, sampler=subsetSampler)

    for batch in dataloader:
        # images, paths, filenames = zip(*batch)
        for image, path, filename in batch:
            filename_stem = Path(filename).stem
            filename_suffix = Path(filename).suffix
            for factory_name in AGENT_FACTORY_REGISTRY.keys():
                for agent in AGENT_FACTORY_REGISTRY.get(factory_name).create_agents():
                    image_clone = image.copy()
                    transformed_image, label = agent.transform(image_clone)
                    TestingUtils.save_image_to_path(transformed_image, Path.cwd() / Path(
                        config['dev']['temp_outout_dir']) / f"{filename_stem}_{label}{filename_suffix}")


if __name__ == '__main__':
    main()