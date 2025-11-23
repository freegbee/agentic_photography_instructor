# import all modules in this directory
from . import color_adjustment
from . import cropping
from . import image_adjustment
from .PairGenerator import generate_transformer_pairs
from .color_adjustment import *
from .image_adjustment import (
    ShiftColorPlaneLeftB,
    ShiftColorPlaneLeftG,
    ShiftColorPlaneLeftR,
    ShiftColorPlaneDownB,
    ShiftColorPlaneDownG,
    ShiftColorPlaneDownR,
    ShiftColorPlaneLeftAll,
    ShiftColorPlaneDownAll,
)

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

__all__ = ["generate_transformer_pairs", "REVERSIBLE_TRANSFORMERS"]
