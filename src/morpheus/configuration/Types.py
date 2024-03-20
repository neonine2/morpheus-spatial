from enum import Enum


class ColName(Enum):
    image_id = "ImageNumber"
    patient_id = "PatientID"
    patch_id = "patch_index"


class CellType(Enum):
    cd8 = "Tcytotoxic"
    tumor = "Tumor"


class Splits(Enum):
    test = "test"
    train = "train"
    validate = "validate"
