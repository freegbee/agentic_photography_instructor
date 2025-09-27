from typing import Dict, List

from transformation_agent.AbstractTransformationAgentFactory import AbstractTransformationAgentFactory
from transformation_agent.TransformationAgent import TransformationAgent
from transformer.AbstractTransformer import AbstractTransformer
from transformer.TransformerTypes import TransformerTypeEnum
from utils.Registries import init_transformer_registry, TRANSFORMER_REGISTRY


class CropTunePixelMatrixAgentFactory(AbstractTransformationAgentFactory):
    """
    Thy factory dynamically creates a set of transformation agents that apply a pattern of available tranformers
    """
    factory_name = "croptune"

    def __init__(self):
        super().__init__()
        init_transformer_registry()

        # prepare hashmap of available transformers
        # for each type of transformer there will be a list of available transformers
        # create agents will dynamically create a crossproduct list of agents with a well defined order
        # e.g. Crop -> ColorAdjust -> Blur

        self.type_map: Dict[TransformerTypeEnum, List[AbstractTransformer]] = {
            TransformerTypeEnum.CROP : [],
            TransformerTypeEnum.IMAGE_ADJUSTMENT: [],
            TransformerTypeEnum.COLOR_ADJUSTMENT : [],
        }

        for transformer_label in TRANSFORMER_REGISTRY.keys():
            transformer = TRANSFORMER_REGISTRY.get(transformer_label)
            if transformer.transformerType in self.type_map:
                self.type_map[transformer.transformerType].append(transformer)



    def _create_agents_impl(self) -> List[TransformationAgent]:
        # Create the transformation agents based on the available transformers
        # Follow a strict order of types
        result: List[TransformationAgent] = []
        for crop_agent in self.type_map[TransformerTypeEnum.CROP]:
            for color_agent in self.type_map[TransformerTypeEnum.COLOR_ADJUSTMENT]:
                for image_agent in self.type_map[TransformerTypeEnum.IMAGE_ADJUSTMENT]:
                    result.append(TransformationAgent(transformer_labels=[crop_agent.label,color_agent.label,image_agent.label]))
        print(f"Created {len(result)} transformation agents")
        return result