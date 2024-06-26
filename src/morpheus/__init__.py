import os
import warnings
from .classification import PatchClassifier, train, load_model, test_model
from .classification.threshold import optimize_threshold
from .datasets.spatial_dataset import SpatialDataset
from .counterfactual.generate import get_counterfactual
from importlib.metadata import version, PackageNotFoundError

try:
    __version__ = version("morpheus-spatial")
except PackageNotFoundError:
    # package is not installed
    __version__ = "unknown"




def check_openmp_conflict():
    if 'KMP_DUPLICATE_LIB_OK' not in os.environ:
        warnings.warn(
            "Potential OpenMP library conflict detected. "
            "If you encounter issues, try setting the environment variable "
            "KMP_DUPLICATE_LIB_OK=TRUE before running your script. "
            "See the package README for more information.",
            RuntimeWarning
        )

check_openmp_conflict()
