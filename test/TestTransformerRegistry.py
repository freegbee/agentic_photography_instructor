import transformer.color_adjustment
import transformer.cropping

from utils.Registries import TRANSFORMER_REGISTRY

if __name__ == "__main__":
    subclasses = list(TRANSFORMER_REGISTRY.keys())
    print(subclasses)
