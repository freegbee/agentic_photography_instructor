import os
from typing import Dict

from torch.utils.data import DataLoader, Dataset, RandomSampler, SequentialSampler, Subset

from image_aquisition.BasicTestDataset import BasicTestDataset
from utils.ConfigLoader import ConfigLoader

def collate_keep_size(batch):
    # Returns a batch without stacking the images, so that they can keep their original size
    return batch


def main():
    config: Dict = ConfigLoader().load(env=os.environ["ENV_NAME"])
    image_dir = config["dev"]["cloned_image_dir"]
    dataset: Dataset = BasicTestDataset(image_dir)
    print(f"Number of images in {image_dir}: {len(dataset)}")

    # sampler to return only 10 random samples from the dataset. Not sure what replacement=True means
    rndSampler = RandomSampler(dataset, replacement=True, num_samples=10)
    subsetSampler = SequentialSampler(Subset(dataset, range(num_samples)))

    # collate_keep_size sorgt dafür, dass die Bilder ihre Originalgrösse behalten können. Standatdmässig müssten alle Bilder die gleiche Grösse haben, was wir aber nicht wollen, da wir hin diesem Moment die Bilder untransformiert lassen wollen.
    dataloader = DataLoader(dataset, batch_size=2, collate_fn=collate_keep_size, sampler=subsetSampler)

    for batch in dataloader:
        # images, paths, filenames = zip(*batch)
        for image, path, filename in batch:
            print(f"path={path}, filenames={filename}, image_shape={image.shape}")


if __name__ == "__main__":
    main()
