from transformation_agent.AbstractTransformationAgentFactory import AbstractTransformationAgentFactory
from transformation_agent.TransformationAgent import TransformationAgent
from transformer.color_adjustment import GrayscaleTransformer
from transformer.cropping import CenterSquareCropTransformer
from transformer.image_adjustment import MedianBlurTransformer3, MedianBlurTransformer5, \
    MedianBlurTransformer7, MedianBlurTransformer9, MedianBlurTransformer11, GaussianBlurTransformer3, \
    GaussianBlurTransformer5, GaussianBlurTransformer7, GaussianBlurTransformer9, GaussianBlurTransformer11, \
    BoxBlurTransformer3, BoxBlurTransformer5, BoxBlurTransformer7, BoxBlurTransformer9, BoxBlurTransformer11


class StaticTransformationAgentFactory(AbstractTransformationAgentFactory):
    factory_name = "static"

    def _create_agents_impl(self):
        return [TransformationAgent([CenterSquareCropTransformer.label, GrayscaleTransformer.label]),
                TransformationAgent([CenterSquareCropTransformer.label]),
                TransformationAgent([GrayscaleTransformer.label]),
                TransformationAgent([MedianBlurTransformer3.label]),
                TransformationAgent([MedianBlurTransformer5.label]),
                TransformationAgent([MedianBlurTransformer7.label]),
                TransformationAgent([MedianBlurTransformer9.label]),
                TransformationAgent([MedianBlurTransformer11.label]),
                TransformationAgent([GaussianBlurTransformer3.label]),
                TransformationAgent([GaussianBlurTransformer5.label]),
                TransformationAgent([GaussianBlurTransformer7.label]),
                TransformationAgent([GaussianBlurTransformer9.label]),
                TransformationAgent([GaussianBlurTransformer11.label]),
                TransformationAgent([BoxBlurTransformer3.label]),
                TransformationAgent([BoxBlurTransformer5.label]),
                TransformationAgent([BoxBlurTransformer7.label]),
                TransformationAgent([BoxBlurTransformer9.label]),
                TransformationAgent([BoxBlurTransformer11.label]),]

