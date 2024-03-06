from enum import Enum

class colname(Enum):
    image_id = 'ImageNumber'
    patient_id = 'PatientID'
    patch_id = 'patch_index'

class celltype(Enum):
    cd8 = 'Tcytotoxic'

class splits(Enum):
    test = 'test'
    train = 'train'
    validate = 'validate'
