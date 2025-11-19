from torch.utils.data import Sampler

from dataset.COCODataset import COCODataset


class TopKSampler(Sampler):
    def __init__(self, data_source: COCODataset, k: int):
        self.data_source = data_source
        self.k = int(k)
        scores = getattr(data_source, "scores", None)
        if scores is None:
            raise ValueError("data_source muss `scores` bereitstellen (z.B. dataset.scores).")
        # highest first
        self.indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:self.k]

    def __iter__(self):
        return iter(self.indices)

    def __len__(self):
        return len(self.indices)