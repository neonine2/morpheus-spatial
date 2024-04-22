from unittest import TestCase

import pytest
import pandas as pd
import numpy as np
from morpheus.utils.patchify import generate_patches_optimized, filter_dataframe, calculate_image_dimensions, \
    calculate_num_patches, calculate_patch_boundaries, create_patch_indices, create_patch_metadata, \
    convert_to_numpy_array, filter_cells_in_patch, create_patch_array, fill_patch_array, update_patch_metadata


@pytest.fixture
def sample_dataframe():
    # Create a sample DataFrame for testing
    data = {
        'ImageNumber': [1, 1, 2, 2],
        'PatientID': [101, 101, 102, 102],
        'Location_Center_X': [100, 200, 300, 400],
        'Location_Center_Y': [150, 250, 350, 450],
        'CellType': ['Type1', 'Type2', 'Type1', 'Type2'],
        'Molecule1': [10, 20, 30, 40],
        'Molecule2': [50, 60, 70, 80]
    }
    return pd.DataFrame(data)


@pytest.fixture
def sample_dataframe_except():
    # Create a sample DataFrame for testing
    data = {
        'ImageNumber': [1, 1, 2, 2],
        'PatientIDX': [101, 101, 102, 102],
        'Location_Center_XX': [100, 200, 300, 400],
        'Location_Center_Y': [150, 250, 350, 450],
        'CellType': ['Type1', 'Type2', 'Type1', 'Type2'],
        'Molecule1': [10, 20, 30, 40],
        'Molecule2': [50, 60, 70, 80]
    }
    return pd.DataFrame(data)


def test_filter_dataframe(sample_dataframe):
    # Test filtering by ImageNumber and PatientID
    filtered_df = filter_dataframe(sample_dataframe, 1, 101)
    assert len(filtered_df) == 2  # Expecting 2 rows as per setup


def test_calculate_image_dimensions(sample_dataframe):
    # Assuming pixel_size of 1 for simplicity
    min_x, min_y, max_x, max_y = calculate_image_dimensions(sample_dataframe, 1)
    assert min_x == 100 and max_x == 401
    assert min_y == 150 and max_y == 451


def test_calculate_num_patches():
    # Simple grid calculation test
    num_patches_x, num_patches_y = calculate_num_patches(0, 0, 400, 400, 100)
    assert num_patches_x == 4 and num_patches_y == 4


def test_calculate_patch_boundaries():
    # Check the boundary calculations
    patch_indices = np.array([[0, 0], [1, 1]])
    start_x, end_x, start_y, end_y = calculate_patch_boundaries(patch_indices, 0, 0, 100)
    assert np.array_equal(start_x, [0, 100])
    assert np.array_equal(end_x, [100, 200])
    assert np.array_equal(start_y, [0, 100])
    assert np.array_equal(end_y, [100, 200])


def test_create_patch_indices():
    # Check indices generation for a 2x2 grid
    indices = create_patch_indices(2, 2)
    assert indices.shape == (4, 2)


def test_create_patch_metadata(sample_dataframe):
    # Testing metadata creation with specific cell types
    patch_indices = np.array([[0, 0], [1, 1]])
    metadata = create_patch_metadata(1, 101, patch_indices, ['Type1', 'Type2'])
    assert 'Contains_Type1' in metadata.columns and 'Contains_Type2' in metadata.columns


def test_convert_to_numpy_array(sample_dataframe):
    # Test conversion to numpy array including selected molecule columns
    array = convert_to_numpy_array(sample_dataframe, ['Molecule1', 'Molecule2'])
    assert array.shape == (4, 5)  # 2 molecule columns + 3 additional


def test_filter_cells_in_patch():
    # Assuming cells are spaced out every 100 pixels
    image_array = np.array([
        [50, 50, 1, 10, 15],
        [150, 150, 2, 20, 25],
        [250, 250, 3, 30, 35],
        [350, 350, 4, 40, 45]
    ])
    filtered_cells = filter_cells_in_patch(image_array, 100, 200, 100, 200, 1)
    assert len(filtered_cells) == 1


def test_create_patch_array():
    # Test the creation of an empty patch array
    patch_array = create_patch_array(100, 2)
    assert patch_array.shape == (100, 100, 2)


def test_fill_patch_array():
    # Test filling the patch array
    patch_array = np.zeros((100, 100, 2))
    patch_cells = np.array([[100, 100, 1, 10, 20]])
    filled_array = fill_patch_array(patch_array, patch_cells, 0, 0, 1, None)
    assert np.sum(filled_array) == 0


def test_update_patch_metadata():
    # Test updating metadata based on cell types present in a patch
    metadata = pd.DataFrame({'Contains_Type1': [False], 'Contains_Type2': [False]})
    update_patch_metadata(metadata, 0, None, ['Type1'], ['Type1', 'Type2'])
    value = metadata.at[0, 'Contains_Type1']
    assert metadata.at[0, 'Contains_Type1']


def test_filter_dataframe_invalid_columns(sample_dataframe_except):
    with pytest.raises(KeyError):
        # Attempt to filter using a non-existent column
        filter_dataframe(sample_dataframe_except, 'ImageNumberX', 'PatientIDX')


def test_calculate_image_dimensions_invalid_column(sample_dataframe_except):
    with pytest.raises(KeyError):
        # Assume a missing required column for calculation
        calculate_image_dimensions(sample_dataframe_except, 1)


def test_empty_dataframe():
    empty_df = pd.DataFrame()
    # Should handle empty data gracefully
    with pytest.raises(KeyError):
        result = filter_dataframe(empty_df, 1, 101)


def test_maximum_patch_dimensions(sample_dataframe):
    # Test the boundary calculation when all points are at the edge of the maximum integer limit
    sample_dataframe['Location_Center_X'] = [2147483647, 2147483646, 100, 100]
    sample_dataframe['Location_Center_Y'] = [2147483647, 2147483646, 100, 100]
    min_x, min_y, max_x, max_y = calculate_image_dimensions(sample_dataframe, 1)
    assert min_x == 100 and max_x == 2147483648


def test_patch_indices_at_limits():
    # Test patch index creation at the limit of typical integer ranges
    indices = create_patch_indices(1, 1)
    assert indices.shape == (1, 2)  # Only one patch expected


def test_integration_generate_patches_optimized(sample_dataframe):
    # Simplified test to check integration of components
    patch_size = 100
    pixel_size = 1
    cell_types = ['Type1', 'Type2']
    molecule_columns = ['Molecule1', 'Molecule2']

    patches, metadata = generate_patches_optimized(
        sample_dataframe, patch_size, pixel_size, cell_types, molecule_columns
    )

    # Check if patches and metadata have expected properties
    assert isinstance(patches, np.ndarray)
    assert isinstance(metadata, pd.DataFrame)
    assert not metadata.empty
