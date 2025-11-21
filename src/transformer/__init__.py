# import all modules in this directory
from . import color_adjustment
from . import cropping
from . import image_adjustment
from .color_adjustment import *

REVERSIBLE_TRANSFORMERS = [SwapColorChannelTransformerGB.label,
                           SwapColorChannelTransformerRB.label,
                           SwapColorChannelTransformerRG.label,
                           InvertColorChannelTransformerB.label,
                           InvertColorChannelTransformerG.label,
                           InvertColorChannelTransformerR.label,
                           InvertColorChannelTransformerBG.label,
                           InvertColorChannelTransformerBR.label,
                           InvertColorChannelTransformerGR.label,
                           InvertColorChannelTransformerAll.label]
