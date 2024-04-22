import json
import numpy as np
import pytest

from morpheus.utils.saving import NumpyEncoder


def test_numpy_integers():
    # Test all supported NumPy integer types
    int_types = [
        np.int_, np.intc, np.intp, np.int8, np.int16, np.int32, np.int64,
        np.uint8, np.uint16, np.uint32, np.uint64
    ]
    for int_type in int_types:
        num = int_type(10)
        result = json.dumps(num, cls=NumpyEncoder)
        assert result == '10'


def test_numpy_floats():
    # Test all supported NumPy float types
    float_types = [np.float_, np.float16, np.float32, np.float64]
    for float_type in float_types:
        num = float_type(10.5)
        result = json.dumps(num, cls=NumpyEncoder)
        assert result == '10.5'


def test_numpy_array():
    # Test encoding of NumPy arrays
    arr = np.array([1, 2, 3])
    result = json.dumps(arr, cls=NumpyEncoder)
    assert result == '[1, 2, 3]'


def test_other_types():
    # Test other types that should be handled by the default JSONEncoder
    data = {"key": "value"}
    result = json.dumps(data, cls=NumpyEncoder)
    assert result == '{"key": "value"}'

    # Test a type that JSONEncoder cannot handle naturally
    with pytest.raises(TypeError):
        json.dumps({1, 2, 3}, cls=NumpyEncoder)
