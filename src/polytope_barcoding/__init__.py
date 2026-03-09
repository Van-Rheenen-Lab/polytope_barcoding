from .core import CellData, CellSegmentation, Binarization, Barcoding, Clustering
from .barcoding import GreedyBarcoding
from .binarization import (
    OtsuBinarize,
    GMMTripletBinarize,
)

__all__ = [
    "CellData",
    "CellSegmentation",
    "Binarization",
    "Barcoding",
    "Clustering",
    "GreedyBarcoding",
    "OtsuBinarize",
    "GMMTripletBinarize",
]

# Optional utilities pull in heavy/GUI dependencies, so keep them lazy.
try:
    from .utils.visualisation_tools import plot_masks

    __all__.append("plot_masks")
except Exception:
    pass

try:
    from .utils.manual_annotation_tools import Annotator, InteractiveThresholding

    __all__.extend(["Annotator", "InteractiveThresholding"])
except Exception:
    pass
