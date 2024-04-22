import pytest
from unittest.mock import patch
import numpy as np
import pandas as pd

from morpheus import SpatialDataset


# Setup a fixture for the dataset
@pytest.fixture
def setup_spatial_dataset(tmp_path):
    # Create a mock CSV file
    csv_path = tmp_path / "input.csv"
    data = pd.DataFrame({
        'patient_id': [1, 2],
        'image_id': [1, 1],
        'cell_type': ['type1', 'type2'],
        'cell_x': [100, 200],
        'cell_y': [150, 250],
        'channel1': [0.1, 0.2]
    })
    data.to_csv(csv_path, index=False)
    return str(csv_path)


def test_initialization(setup_spatial_dataset):
    dataset = SpatialDataset(input_path=setup_spatial_dataset)
    assert dataset.input_path == setup_spatial_dataset


def test_load_input_csv(setup_spatial_dataset):
    dataset = SpatialDataset(input_path=setup_spatial_dataset)
    data = dataset.load_input_csv()
    assert not data.empty
    assert 'patient_id' in data.columns


def test_check_input_csv(setup_spatial_dataset):
    dataset = SpatialDataset(input_path=setup_spatial_dataset)
    with pytest.raises(ValueError):
        dataset.check_input_csv(pd.DataFrame(), [])


def test_generate_masked_patch(setup_spatial_dataset):
    dataset = SpatialDataset(input_path=setup_spatial_dataset)
    with patch('morpheus.utils.patchify.generate_patches_optimized',
               return_value=(np.random.rand(10, 16, 16, 1), pd.DataFrame({'index': range(10)}))) as mock_gen:
        dataset.generate_masked_patch(cell_to_mask=['type1'], save=False)
        assert mock_gen.called


def test_load_patch_data_error(setup_spatial_dataset):
    dataset = SpatialDataset(input_path=setup_spatial_dataset, patch_path='invalid_path.h5')
    with pytest.raises(Exception):
        dataset.load_patch_data()


def test_get_split_ids_error(setup_spatial_dataset):
    dataset = SpatialDataset(input_path=setup_spatial_dataset, split_dir='non_existent_directory')
    with pytest.raises(Exception):
        dataset.get_split_ids('train')


def test_generate_data_splits(setup_spatial_dataset):
    dataset = SpatialDataset(input_path=setup_spatial_dataset)
    with patch.object(dataset, 'get_patient_splits', return_value=None):
        with pytest.raises(ValueError):
            dataset.generate_data_splits(stratify_by='cell_type')


def test_set_patch_path(setup_spatial_dataset):
    dataset = SpatialDataset(input_path=setup_spatial_dataset)
    assert dataset.set_patch_path(None) is None


def test_check_loaded_patch(setup_spatial_dataset):
    dataset = SpatialDataset(input_path=setup_spatial_dataset)
    dataset.metadata = pd.DataFrame({'patch_id': [1, 2]})
    dataset.channel_names = ['channel1']
    dataset.n_channels = 1
    dataset.check_loaded_patch()  # Should not raise
