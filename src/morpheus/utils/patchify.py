import pandas as pd
import numpy as np
import h5py
from ..configuration.Types import ColName

from tqdm import tqdm

def save_masked_patch_to_hdf5(
    df, patch_size, pixel_size, cell_types, molecule_columns: list, mask_cell_types=None, save_path="patch.h5"
):
    # Reorder the molecule_columns to match the order in the input DataFrame
    molecule_columns = [col for col in df.columns if col in molecule_columns]
    num_channels = len(molecule_columns)

    # Get data types for first molecule columns
    data_type = df[molecule_columns].iloc[:, 0].dtypes

    # Get all combinations of image numbers and patient IDs in the DataFrame
    all_image = df[[ColName.image_id.value, ColName.patient_id.value]].drop_duplicates()

    metadata = []
    with h5py.File(save_path, "w") as f:
        # Create a dataset to store the images
        dset = f.create_dataset(
            "images",
            shape=(0, patch_size, patch_size, num_channels),
            maxshape=(None, patch_size, patch_size, num_channels),
            compression="gzip",
            chunks=(100, patch_size, patch_size, num_channels),
            dtype=data_type,
        )

        for _, image in tqdm(all_image.iterrows(), total=len(all_image)):
            # Filter the DataFrame for the current image and patient
            image_number, patient_id = (
                image[ColName.image_id.value],
                image[ColName.patient_id.value],
            )
            image_df = filter_dataframe(df, image_number, patient_id)

            min_x, min_y, max_x, max_y = calculate_image_dimensions(image_df, pixel_size)
            num_patches_x, num_patches_y = calculate_num_patches(
                min_x, min_y, max_x, max_y, patch_size
            )
            patch_indices = create_patch_indices(num_patches_x, num_patches_y)
            start_x, end_x, start_y, end_y = calculate_patch_boundaries(
                patch_indices, min_x, min_y, patch_size
            )

            patch_metadata = create_patch_metadata(
                image_number, patient_id, patch_indices, cell_types
            )
            image_array = convert_to_numpy_array(image_df, molecule_columns)

            patches_list = []
            for i in range(len(patch_indices)):
                patch_cells = filter_cells_in_patch(
                    image_array, start_x[i], end_x[i], start_y[i], end_y[i], pixel_size
                )
                # if len(patch_cells) == 0:  # Skip empty patches
                # continue
                patch_array = create_patch_array(patch_size, num_channels)
                patch_array = fill_patch_array(
                    patch_array,
                    patch_cells,
                    start_x[i],
                    start_y[i],
                    pixel_size,
                    mask_cell_types,
                )

                # append the patch to the list of patches
                patches_list.append(patch_array)

                update_patch_metadata(
                    patch_metadata,
                    i,
                    image_df,
                    patch_cells[:, image_df.columns.get_loc(ColName.cell_type.value)],
                    cell_types,
                )

            # concatenate the patches and save them to the HDF5 file
            patches_array = np.stack(patches_list)
            num_patches = patches_array.shape[0]

            # Resize the dataset to accommodate new patches
            dset.resize(dset.shape[0] + num_patches, axis=0)

            # Write all patches for this image at once to the end of the dataset
            dset[-num_patches:] = patches_array

            metadata.append(patch_metadata)
        total_patches = dset.shape[0]
    metadata_df = pd.concat(metadata, ignore_index=True)

    if total_patches != len(metadata_df):
        raise ValueError("Number of patches generated does not match size of metadata")
    
    return metadata_df


def filter_dataframe(df, image_number, patient_id):
    return df[
        (df[ColName.image_id.value] == image_number)
        & (df[ColName.patient_id.value] == patient_id)
    ]


def calculate_image_dimensions(image_df, pixel_size):
    min_x = int(image_df[ColName.cell_x.value].min() // pixel_size)
    min_y = int(image_df[ColName.cell_y.value].min() // pixel_size)
    max_x = int(image_df[ColName.cell_x.value].max() // pixel_size) + 1
    max_y = int(image_df[ColName.cell_y.value].max() // pixel_size) + 1
    return min_x, min_y, max_x, max_y


def calculate_num_patches(min_x, min_y, max_x, max_y, patch_size):
    num_patches_x = (max_x - min_x) // patch_size
    num_patches_y = (max_y - min_y) // patch_size
    return num_patches_x, num_patches_y


def calculate_patch_boundaries(patch_indices, min_x, min_y, patch_size):
    start_x = patch_indices[:, 0] * patch_size + min_x
    end_x = start_x + patch_size
    start_y = patch_indices[:, 1] * patch_size + min_y
    end_y = start_y + patch_size
    return start_x, end_x, start_y, end_y


def create_patch_indices(num_patches_x, num_patches_y):
    return np.mgrid[0:num_patches_x, 0:num_patches_y].reshape(2, -1).T


def create_patch_metadata(image_number, patient_id, patch_indices, cell_types):
    patch_metadata = pd.DataFrame(
        {
            ColName.image_id.value: image_number,
            ColName.patient_id.value: patient_id,
            "PatchIndex_X": patch_indices[:, 0],
            "PatchIndex_Y": patch_indices[:, 1],
        }
    )
    for cell_type in cell_types:
        patch_metadata[f"Contains_{cell_type}"] = False
    return patch_metadata


def convert_to_numpy_array(image_df, molecule_columns):
    return image_df[
        [ColName.cell_x.value, ColName.cell_y.value, ColName.cell_type.value]
        + molecule_columns
    ].values


def filter_cells_in_patch(image_array, start_x, end_x, start_y, end_y, pixel_size):
    mask = (
        (image_array[:, 0] >= start_x * pixel_size)
        & (image_array[:, 0] < end_x * pixel_size)
        & (image_array[:, 1] >= start_y * pixel_size)
        & (image_array[:, 1] < end_y * pixel_size)
    )
    return image_array[mask]


def create_patch_array(patch_size, num_channels):
    return np.zeros((patch_size, patch_size, num_channels))


def fill_patch_array(
    patch_array, patch_cells, start_x, start_y, pixel_size, mask_cell_types
):
    x_indices = ((patch_cells[:, 0] - start_x * pixel_size) // pixel_size).astype(int)
    y_indices = ((patch_cells[:, 1] - start_y * pixel_size) // pixel_size).astype(int)

    if mask_cell_types is not None:
        mask_indices = np.isin(patch_cells[:, 2], mask_cell_types)
        patch_cells = (
            patch_cells.copy()
        )  # Create a copy to avoid modifying the original array
        patch_cells[mask_indices, 3:] = 0

    # Ensure that x_indices and y_indices are within the valid range
    valid_indices = (
        (x_indices >= 0)
        & (x_indices < patch_array.shape[1])
        & (y_indices >= 0)
        & (y_indices < patch_array.shape[0])
    )
    x_indices = x_indices[valid_indices]
    y_indices = y_indices[valid_indices]
    patch_cells = patch_cells[valid_indices]

    np.add.at(patch_array, (y_indices, x_indices), patch_cells[:, 3:])

    return patch_array


def update_patch_metadata(patch_metadata, i, image_df, patch_cell_types, cell_types):
    for cell_type in cell_types:
        if cell_type in patch_cell_types:
            patch_metadata.at[i, f"Contains_{cell_type}"] = True
