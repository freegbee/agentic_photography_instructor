from typing import Dict, Tuple, List


class SplitRatios:
    def __init__(self):
        self.ratios_dict: Dict[str, float] = {}

    def __str__(self):
        return str(self.ratios_dict)

    def __repr__(self):
        return f"SplitRatios({self.ratios_dict})"

    def add_ratio(self, split_name: str, ratio: float):
        self.ratios_dict[split_name] = ratio

    def with_ratio(self, split_name: str, ratio: float) -> 'SplitRatios':
        self.ratios_dict[split_name] = ratio
        return self

    def get_ratios(self) -> dict[str, float]:
        return self.ratios_dict

    def get_default_ratios(self) -> Tuple[float, float, float]:
        return self.ratios_dict['train'], self.ratios_dict['validation'], self.ratios_dict['test']

    def get_split_indices(self, indices: List[int]) -> Dict[str, List[int]]:
        result: Dict[str, List[int]] = {}
        current_position = 0
        total_size = len(indices)
        items = list(self.ratios_dict.items())
        for i, (split_name, ratio) in enumerate(items):
            if i == len(items) - 1:
                # Last entry, take all remaining
                result[split_name] = indices[current_position:]
                current_position = total_size
            else:
                # calculate length and cap it to not overrun
                ratio_length = int(total_size * ratio)
                remaining = total_size - current_position
                take = min(ratio_length, remaining)
                result[split_name] = indices[current_position: current_position + take]
                current_position += take
        return result

    @staticmethod
    def create_default_split_ratios() -> 'SplitRatios':
        return SplitRatios.create_with_ratios(0.6, 0.2, 0.2)

    @staticmethod
    def create_with_ratios(train_ratio: float, val_ratio: float, test_ratio: float) -> 'SplitRatios':
        split_ratios = SplitRatios()
        split_ratios.add_ratio("train", train_ratio)
        split_ratios.add_ratio("validation", val_ratio)
        split_ratios.add_ratio("test", test_ratio)
        return split_ratios
