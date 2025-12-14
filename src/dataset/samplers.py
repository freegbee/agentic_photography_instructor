from typing import List, Iterator

from torch.utils.data import Sampler


class IndexSampler(Sampler[int]):
    def __init__(self, indices: List[int]):
        super().__init__()
        self.indices = indices

    def __iter__(self) -> Iterator[int]:
        yield from self.indices

    def __len__(self) -> int:
        return len(self.indices)