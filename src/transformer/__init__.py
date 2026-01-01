# import all modules in this directory
from . import color_adjustment
from . import cropping
from . import image_adjustment
from . import lighting
from .PairGenerator import generate_transformer_pairs
from .color_adjustment import *
from .image_adjustment import *
from .lighting import *

REVERSIBLE_TRANSFORMERS = [SwapColorChannelTransformerGB.label,
                           SwapColorChannelTransformerRB.label,
                           SwapColorChannelTransformerRG.label,
                           InvertColorChannelTransformerB.label,
                           InvertColorChannelTransformerG.label,
                           InvertColorChannelTransformerR.label,
                           ShiftColorPlaneLeftB.label,
                           ShiftColorPlaneLeftG.label,
                           ShiftColorPlaneLeftR.label,
                           ShiftColorPlaneDownB.label,
                           ShiftColorPlaneDownG.label,
                           ShiftColorPlaneDownR.label,
                           ShiftColorPlaneLeftAll.label,
                           ShiftColorPlaneDownAll.label,
                           ]

POC_ONE_WAY_TRANSFORMERS = [
    ShiftColorPlaneDownAll.label
]

POC_TWO_WAY_TRANSFORMERS = [
    ShiftColorPlaneDownAll.label,
    InvertColorChannelTransformerR.label
]

POC_MULTI_ONE_STEP_TRANSFORMERS = [
    InvertColorChannelTransformerB.label,
    InvertColorChannelTransformerG.label,
    InvertColorChannelTransformerR.label,
    SwapColorChannelTransformerRB.label,
    SwapColorChannelTransformerRG.label,
    SwapColorChannelTransformerGB.label
]

SENSIBLE_TRANSFORMERS = color_adjustment.SENSIBLE_CA_TRANSFORMERS + image_adjustment.SENSIBLE_IA_TRANSFORMERS + cropping.SENSIBLE_CROP_TRANSFORMERS + lighting.SENSIBLE_LIGHTING_TRANSFORMERS

__all__ = ["generate_transformer_pairs", "REVERSIBLE_TRANSFORMERS", "POC_ONE_WAY_TRANSFORMERS", "POC_TWO_WAY_TRANSFORMERS", "POC_MULTI_ONE_STEP_TRANSFORMERS"]
