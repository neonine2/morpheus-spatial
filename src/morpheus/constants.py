from enum import Enum

class colname(Enum):
    IMAGEID = 'ImageNumber'
    PATIENTID = 'PatientID'

class celltype(Enum):
    cd8 = 'Tcytotoxic'

class splits(Enum):
    test = 'test'
    train = 'train'
    validate = 'validate'
