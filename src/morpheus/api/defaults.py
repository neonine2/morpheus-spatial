"""
This module defines the default metadata and data dictionaries.
Note that the "name" field is automatically populated upon initialization
"""

DEFAULT_META = {
    "name": None,
    "type": ["blackbox", "tensorflow", "keras"],
    "explanations": ["local"],
    "params": {},
    "version": None,
}  # type: dict


DEFAULT_DATA = {
    "cf": None,
    "all": [],
    "orig_class": None,
    "orig_proba": None,
    "id_proto": None,
}  # type: dict
