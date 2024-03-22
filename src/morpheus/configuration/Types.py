from enum import Enum


class ColName(Enum):
    image_id = "ImageNumber"
    patient_id = "PatientID"
    cell_type = "CellType"
    cell_x = "Location_Center_X"
    cell_y = "Location_Center_Y"
    patch_id = "patch_index"
    splits = "splits"


class CellType(Enum):
    cd8 = "Tcytotoxic"
    tumor = "Tumor"


class Splits(Enum):
    test = "test"
    train = "train"
    validate = "validate"


class DefaultFolderName(Enum):
    split = "split"
    model = "model"
    counterfactual = "cf"


class DefaultFileName(Enum):
    label = "label.csv"
    kdtree = "kdtree.pkl"
