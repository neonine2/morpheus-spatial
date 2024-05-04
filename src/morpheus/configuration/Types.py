from enum import Enum


class ColName(Enum):
    image_id = "ImageNumber"
    patient_id = "PatientID"
    cell_type = "CellType"
    cell_x = "Location_Center_X"
    cell_y = "Location_Center_Y"
    patch_id = "patch_id"
    splits = "splits"
    contains_tumor = "Contains_Tumor"
    contains_cd8 = "Contains_Tcytotoxic"


class CellType(Enum):
    cd8 = "Tcytotoxic"
    tumor = "Tumor"


class Splits(Enum):
    train = "train"
    validate = "validate"
    test = "test"


class DefaultFolderName(Enum):
    split = "split"
    model = "model"
    counterfactual = "cf"


class DefaultFileName(Enum):
    label = "label.csv"
    kdtree = "kdtree.pkl"
    patch = "patch.h5"
    normalization = "normalization_params.json"
