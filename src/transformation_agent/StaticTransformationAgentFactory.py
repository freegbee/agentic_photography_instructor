from transformation_agent.AbstractTransformationAgentFactory import AbstractTransformationAgentFactory
from transformation_agent.TransformationAgent import TransformationAgent
from transformer.color_adjustment import GrayscaleTransformer
from transformer.cropping import CenterSquareCropTransformer


class StaticTransformationAgentFactory(AbstractTransformationAgentFactory):
    factory_name = "static"

    def _create_agents_impl(self):
        return [TransformationAgent([CenterSquareCropTransformer.label, GrayscaleTransformer.label]),
                TransformationAgent([CenterSquareCropTransformer.label]),
                TransformationAgent([GrayscaleTransformer.label])]
