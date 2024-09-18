import json
import os
import warnings

import shutil
import h5py
import numpy as np
import pandas as pd
import ray
from tqdm.auto import trange, tqdm

from ..utils.patchify import save_masked_patch_to_hdf5
from ..configuration.Types import (
    ColName,
    Splits,
    DefaultFolderName,
    DefaultFileName,
)


class SpatialDataset:
    def __init__(
        self,
        input_path: str,
        channel_names: list = [],
        additional_cols: list = [],
        channel_path: str = None,
        patch_path: str = None,
        split_dir: str = None,
        model_path: str = None,
        cf_dir: str = None,
        verbose: bool = False,
    ):
        self.data_dim = None
        self.metadata = None
        self.input_path = input_path
        self.root_dir = os.path.dirname(input_path)

        if channel_path is not None:
            with open(channel_path, "r") as f:
                channel_names = f.read().splitlines()

        self.load_input_csv(
            channel_names,
            additional_cols,
            check_only=True,
            verbose=verbose,
        )  # also sets self.channel_names

        # set the directories where different outputs are saved
        self.patch_path = self.set_patch_path(patch_path)
        self.split_dir = self.set_split_dir(split_dir)
        self.model_path = self.set_model_path(model_path)
        self.cf_dir = self.set_counterfactual_dir(cf_dir)

        if self.patch_path is not None:
            self.load_patch_data()
            self.check_loaded_patch()

        # concatenate split name to metadata if splits available
        if self.split_dir is not None and self.patch_path is not None:
            self.get_split_info()

        # display all the directories set
        if verbose:
            self.display_directories()

    def display_directories(self):
        print(f"Input path: {self.input_path}")
        print(f"Patch path: {self.patch_path}")
        print(f"Split directory: {self.split_dir}")
        print(f"Model path: {self.model_path}")
        print(f"Counterfactual directory: {self.cf_dir}")

    def get_split_info(self):
        for split in Splits:
            self.metadata.loc[self.get_split_ids(split.value), ColName.splits.value] = (
                split.value
            )

    def load_input_csv(
        self,
        channel_names: list = [],
        additional_cols: list = [],
        check_only=False,
        verbose=False,
    ):
        try:
            input_csv = pd.read_csv(self.input_path, low_memory=False)
        except Exception as e:
            print(f"Error loading input CSV: {e}")

        # check that the input CSV is not empty
        if input_csv.empty:
            raise ValueError("Input CSV is empty")

        # check the input CSV contains the required columns
        required_cols = [
            ColName.patient_id.value,
            ColName.image_id.value,
            ColName.cell_type.value,
            ColName.cell_x.value,
            ColName.cell_y.value,
        ]
        if len(channel_names) == 0:  # should only be done during dataset initialization
            channel_names = [
                col
                for col in input_csv.columns
                if col not in required_cols + additional_cols
            ]
            if verbose:
                print(
                    f"{len(channel_names)} channels inferred from input CSV: {channel_names}"
                )
        elif len(additional_cols) == 0:
            additional_cols = [
                col
                for col in input_csv.columns
                if col not in required_cols + channel_names
            ]
        required_cols += channel_names + additional_cols

        for col in required_cols:
            if col not in input_csv.columns:
                warnings.warn("input csv does not contain required column: {col}")

        # reorder the channel names to match the order in the input CSV
        self.channel_names = [col for col in input_csv.columns if col in channel_names]

        if check_only:
            return
        else:
            # reorder columns in the input CSV
            return input_csv[required_cols]

    def generate_masked_patch(
        self,
        cell_to_mask: list = [],
        patch_size: int = 16,
        pixel_size: int = 3,
        cell_types: list = None,
        save_path: str = None,
    ):
        """
        Generate masked patches from the input data.

        Args:
            cell_to_mask (str): The cell type to mask.
            patch_size (int): The size of the patch in pixels.
            pixel_size (int): The pixel size in microns.
            cell_types (list): The cell types to include in the metadata.
            save (bool): Whether to save the patches.
            save_path (str): The path to save the patches.

        Returns:

        """
        # check the save_path is not already present
        self.patch_path = (
            save_path
            if save_path is not None
            else os.path.join(self.root_dir, DefaultFileName.patch.value)
        )
        if os.path.isfile(self.patch_path):
            self.load_patch_data()
            self.check_loaded_patch()
            print(f"File {self.patch_path} already exists, existing file loaded")
            print(f"Total number of patches: {len(self.metadata)}")
            return

        # print out details about the patches
        print(f"Generating patches of size {patch_size}x{patch_size} pixels")
        print(f"Pixel size: {pixel_size}x{pixel_size} microns")
        print(f"Cell types recorded: {cell_types}")
        print(f"Cell types masked: {cell_to_mask}")

        # load the input data
        df = self.load_input_csv(channel_names=self.channel_names)

        # generate the patches
        metadata_df = save_masked_patch_to_hdf5(
            df, patch_size, pixel_size, cell_types, molecule_columns=self.channel_names, mask_cell_types=cell_to_mask, save_path=self.patch_path
        )
        metadata_df = SpatialDataset.convert_object_columns(
            metadata_df.reset_index().rename(columns={"index": ColName.patch_id.value})
        )
        metadata_df = SpatialDataset.convert_id_to_integer(metadata_df, [ColName.patient_id.value, ColName.image_id.value, ColName.patch_id.value])
        self.metadata = metadata_df

        with h5py.File(self.patch_path, "a") as f: # append mode
            # Add channel names
            f.create_dataset("channel_names", data=self.channel_names)

            # Convert the metadata DataFrame to a structured NumPy array
            metadata_numpy = metadata_df.to_records(index=False)

            # Add metadata as a new dataset
            f.create_dataset(
                "metadata", data=metadata_numpy, dtype=metadata_numpy.dtype
            )

        print(f"Patches saved to {self.patch_path}")
        self.load_patch_data()
        self.check_loaded_patch()
        print(f"Number of patches generated: {len(metadata_df)}")
        print(f"Example patch metadata:\n{metadata_df.head()}")
        return

    def set_patch_path(self, patch_path: str = None):
        path = (
            patch_path
            if patch_path is not None
            else os.path.join(self.root_dir, DefaultFileName.patch.value)
        )
        return path if os.path.isfile(path) else None

    def set_split_dir(self, split_dir: str = None):
        dir = (
            split_dir
            if split_dir is not None
            else os.path.join(self.root_dir, DefaultFolderName.split.value)
        )
        return dir if os.path.isdir(dir) else None

    def set_model_path(self, model_path: str = None):
        if model_path is not None:
            if os.path.isfile(model_path):
                return model_path
            else:
                raise ValueError(f"Model file not found at {model_path}")
        else:
            model_dir = os.path.join(self.root_dir, DefaultFolderName.model.value)
            DEFAULT_UNET = os.path.join(model_dir, 'unet.ckpt')
            if os.path.isfile(DEFAULT_UNET):
                return DEFAULT_UNET
            # look into model_dir (and possible subdirectories) for the first model file
            for root, _, files in os.walk(model_dir):
                for file in files:
                    if file.endswith(".ckpt"):
                        return os.path.join(root, file)

    def set_counterfactual_dir(self, cf_dir: str = None):
        dir = (
            cf_dir
            if cf_dir is not None
            else os.path.join(self.root_dir, DefaultFolderName.counterfactual.value)
        )
        return dir if os.path.isdir(dir) else None

    def check_loaded_patch(self):
        # check that the data is loaded
        if not hasattr(self, "metadata"):
            raise ValueError("Metadata not loaded")
        if not hasattr(self, "channel_names"):
            raise ValueError("Channel names not loaded")

        # check that the data is consistent
        if len(self.metadata) != self.metadata[ColName.patch_id.value].nunique():
            raise ValueError("Metadata contains duplicate patch IDs")
        if len(self.channel_names) != self.n_channels:
            raise ValueError("Number of channel names do not match data dimensions")

        # check key metadata columns are present
        for col in [ColName.patient_id.value, ColName.patch_id.value]:
            if col not in self.metadata.columns:
                raise ValueError(f"Metadata missing column: {col}")

    def load_patch_data(self):
        try:
            with h5py.File(self.patch_path, "r") as f:
                self.metadata = pd.DataFrame(f["metadata"][:])
                self.channel_names = [
                    name.decode("utf-8") for name in f["channel_names"][:]
                ]
                data_shape = f["images"].shape
                self.n_patches, self.img_size, self.n_channels = (
                    data_shape[0],
                    data_shape[1:3],
                    data_shape[3],
                )
        except Exception as e:
            print(f"Error loading data: {e}")

        # if patient id is string, decode it with utf-8
        if self.metadata[ColName.patient_id.value].dtype == object:
            self.metadata[ColName.patient_id.value] = self.metadata[
                ColName.patient_id.value
            ].str.decode("utf-8")

    def get_split_ids(self, split_name: str):
        label_path = os.path.join(
            self.split_dir, split_name, DefaultFileName.label.value
        )
        try:
            return pd.read_csv(label_path)[ColName.patch_id.value].values
        except Exception as e:
            print(f"Error loading split {split_name}: {e}")

    def generate_data_splits(
        self,
        stratify_by: str,
        train_size: float = 0.6,
        val_size: float = 0.2,
        test_size: float = 0.2,
        save_dir=None,
        random_state=None,
        shuffle=True,
        tolerance=None,
        specify_split: dict = None,
        save=True,
    ) -> dict:
        """
        Generate train, validation, and test data splits, and save the data splits to the given directory

        Args:
        stratify_by: str
            The column name to stratify the data by
        save_dir: str
            The directory to save the data splits
        train_size: float
            The proportion of the dataset to include in the train split
        val_size: float
            The proportion of the dataset to include in the validation split
        test_size: float
            The proportion of the dataset to include in the test split
        random_state: int
            Controls the shuffling applied to the data before applying the split.
        shuffle: bool
            Whether to shuffle the data before splitting
        tolerance: dict
            A dictionary of tolerance parameters to control the data split generation
            - eps: float
                The tolerance for the difference in proportions between the train and test/validate splits
            - train_lb: float
                The lower bound for the proportion of the train split
            - n_tol: int
                The number of attempts to generate a valid data split
        specify_split: dict
            A dictionary specifying patient IDs in the train, validation, and test splits
        save: bool
            Whether to save the data splits to the save directory
        """
        self.label_name = stratify_by

        if tolerance is None:
            tolerance = {"eps": 0.01, "train_lb": 0.5, "n_tol": 100}
        if save_dir is None:
            self.split_dir = os.path.join(
                self.root_dir, DefaultFolderName.split.value
            )  # default save directory
        else:
            self.split_dir = save_dir

        if os.path.isdir(os.path.join(self.split_dir, Splits.train.value)):
            self.get_split_info()
            print(f"Data splits already exist in {self.split_dir}")
            return

        if specify_split is None:
            patient_split = self.get_patient_splits(
                stratify_by,
                train_size,
                val_size,
                test_size,
                random_state,
                shuffle,
                **tolerance,
            )
        else:
            if SpatialDataset.issplitvalid(specify_split):
                patient_split = {
                    name: np.array(spt) for name, spt in specify_split.items()
                }
            else:
                raise ValueError("Given split is not valid")

        if patient_split is None:
            raise ValueError(
                "Could not satisfy data split constraints, try again or adjust constraints"
            )

        if save:
            try:
                self.save_splits(patient_split, label_name=stratify_by)
                print(f"Data splits saved to {self.split_dir}")
                self.get_split_info()
                return patient_split
            except Exception as e:
                print(f"Error saving splits: {e}")
                # remove the split directory if an error occurs
                if os.path.isdir(self.split_dir):
                    shutil.rmtree(self.split_dir)

    @staticmethod
    def is_valid_split(split_info, eps=0.01, train_lb=0.65):
        tr_prop = split_info[Splits.train.value][0]
        tr_te_diff = abs(
            split_info[Splits.train.value][1] - split_info[Splits.test.value][1]
        )
        tr_va_diff = abs(
            split_info[Splits.train.value][1] - split_info[Splits.validate.value][1]
        )
        return (tr_te_diff < eps) and (tr_va_diff < eps) and (tr_prop > train_lb)

    def get_patient_splits(
        self,
        stratify_by,
        train_size,
        val_size,
        test_size,
        random_state=42,
        shuffle=True,
        n_tol=100,
        eps=0.01,
        train_lb=0.65,
    ):
        assert (
            train_size + val_size + test_size == 1
        ), "train_size, val_size, and test_size should sum to 1"

        patient_id = np.unique(self.metadata[ColName.patient_id.value])
        n_patches = len(self.metadata)
        isValidSplit = False
        counter = 0
        while not isValidSplit and counter < n_tol:
            if shuffle:
                rng = np.random.default_rng(random_state)
                rng.shuffle(patient_id)
            train_end = int(len(patient_id) * train_size)
            valid_end = train_end + int(len(patient_id) * val_size)
            patient_split = {
                Splits.train.value: patient_id[:train_end],
                Splits.validate.value: patient_id[train_end:valid_end],
                Splits.test.value: patient_id[valid_end:],
            }
            # compute the proportion of each split based on specified stratification
            split_info = {}
            for split, pat in patient_split.items():
                data_sub = self.metadata[
                    self.metadata[ColName.patient_id.value].isin(pat)
                ]
                split_info[split] = [
                    len(data_sub) / n_patches,
                    data_sub[stratify_by].mean(),
                ]

            if SpatialDataset.is_valid_split(split_info, eps, train_lb):
                print(
                    "Split constraints satisfied\nPatch proportions and Positive patch proportions:"
                )
                for split, dat in split_info.items():
                    print(f"{split:<10}: {dat[0]:>5.3f}, {dat[1]:>5.3f}")
                return patient_split
            else:
                counter += 1
        return None

    def summarize_split(self):
        pass

    def save_splits(self, patient_split, label_name, chunk_size=2000):
        """
        Save the data splits to the specified directory in a chunked manner to avoid memory issues.

        Args:
            patient_split: dict
                A dictionary containing the patient IDs for the train, validation, and test splits
            label_name: str
                The metadata column to use as the label
            chunk_size: int
                The number of patches to process at a time
        """

        # obtain patch index corresponding to patient split
        split_index = {
            key: self.metadata[self.metadata[ColName.patient_id.value].isin(val)].index.to_numpy(dtype=int)
            for key, val in patient_split.items()
        }

        with h5py.File(self.patch_path, "r") as patch_file:
            _, h, w, n_channels = patch_file["images"].shape  # Assuming channel-last image format

            for split_name, _index in split_index.items():

                # make directories for the split
                _path = os.path.join(self.split_dir, split_name)
                os.makedirs(os.path.join(_path, "0"), exist_ok=True)
                os.makedirs(os.path.join(_path, "1"), exist_ok=True)

                # Initialize an empty list to store metadata for the entire split
                metadata = self.metadata.iloc[_index][
                        [
                            ColName.patch_id.value,
                            label_name,
                            ColName.patient_id.value,
                            ColName.image_id.value,
                        ]
                    ]
                metadata.to_csv(
                    os.path.join(_path, DefaultFileName.label.value), index=False
                )
        
                # Initialize accumulators for normalization                
                mean_accumulator = np.zeros(n_channels)
                sum_square_accumulator = np.zeros(n_channels)

                for i in trange(0, len(_index), chunk_size, desc=f"Saving {split_name} split in chunks"):
                    _indices = _index[i:i+chunk_size]
                    _patches = patch_file["images"][_indices, ...]  # Lazy loading of patches

                    # save patches
                    n_patches = len(_indices)
                    _labels = self.metadata.iloc[_indices][label_name].values.astype(int)
                    _ids = self.metadata.iloc[_indices][ColName.patch_id.value].values
                    for i in trange(n_patches, leave=False):
                        save_path = os.path.join(_path, f"{_labels[i]}/patch_{_ids[i]}.npy")
                        np.save(save_path, _patches[i, ...])

                    # Accumulate for mean and variance calculation
                    if split_name == Splits.train.value:
                        mean_accumulator += np.sum(_patches, axis=(0, 1, 2))
                        sum_square_accumulator += np.sum(_patches ** 2, axis=(0, 1, 2))

                # save normalization parameters
                if split_name == Splits.train.value:
                    # Calculate the final mean and std
                    total_pixel_count = len(_index) * h * w
                    mean = mean_accumulator / total_pixel_count
                    variance = (sum_square_accumulator / total_pixel_count) - (mean ** 2)
                    std = np.sqrt(variance)

                    normalization_params = {    
                        "mean": mean.tolist(),
                        "stdev": std.tolist(),
                    }
                    normalization_path = os.path.join(self.split_dir, "normalization_params.json")
                    with open(normalization_path, "w") as normalization_file:
                        json.dump(normalization_params, normalization_file)
                        print(f"Normalization parameters saved to {normalization_path}")

    @staticmethod
    def convert_id_to_integer(metadata, col_name=[]):
        """
        Convert the columns to integers if they are floats.
        """
        for column in col_name:
            if issubclass(metadata[column].dtype.type, float):
                metadata[column] = metadata[column].astype(int)
        return metadata

    def load_from_metadata(
        self, metadata, col_as_label=ColName.contains_cd8.value, parallel=False
    ):
        """
        Load all images with patch_ids in the list from the dataset.
        """
        # join the label column with the patch_id column to form the image path
        image_paths = metadata.apply(
            lambda x: os.path.join(
                self.split_dir,
                x[ColName.splits.value],
                f"{int(x[col_as_label])}/patch_{x[ColName.patch_id.value]}.npy",
            ),
            axis=1,
        )
        if parallel:
            ray.shutdown()
            ray.init()
            # Launch tasks in parallel
            futures = [parallel_load_image.remote(path) for path in image_paths]

            # Retrieve results
            images = []
            for future in tqdm(
                ray.get(futures), total=len(image_paths), desc="Loading Images"
            ):
                images.append(future)
        else:
            images = [self.load_single_image(path, id=False) for path in image_paths]
        images = np.stack(images, axis=0)
        return images

    @staticmethod
    def load_single_image(path, id=True):
        """
        Load a single image from the dataset.

        Args:
            path (str): Path to the image file.
            id (bool): Whether to return the image ID.

        Returns:
            np.ndarray: Image array.
        """
        image = np.load(path)
        if id:
            return image, int(
                os.path.splitext(os.path.basename(path))[0].split("_")[-1]
            )
        else:
            return image

    def generate_patch_path(self, patch_id, label, split):
        label = int(label)
        return os.path.join(self.split_dir, split, f"{label}/patch_{patch_id}.npy")

    @staticmethod
    def convert_object_columns(df):
        for column in df.columns:
            if df[column].dtype == object:
                try:
                    df[column] = df[column].astype(int)
                except (ValueError, TypeError):
                    df[column] = df[column].astype(
                        h5py.string_dtype(encoding="utf-8", length=255)
                    )
        return df

    @staticmethod
    def issplitvalid(split):
        # check that the given split is valid
        if len(split) != 3:
            raise ValueError(
                "Given split should contain three lists of patient IDs for train, validation, and test splits"
            )
        if len(set(split[Splits.train.value]) & set(split[Splits.validate.value])) > 0:
            raise ValueError("Train and validation splits contain the same patient IDs")
        if len(set(split[Splits.train.value]) & set(split[Splits.test.value])) > 0:
            raise ValueError("Train and test splits contain the same patient IDs")
        if len(set(split[Splits.validate.value]) & set(split[Splits.test.value])) > 0:
            raise ValueError("Validation and test splits contain the same patient IDs")
        return True



@ray.remote
def parallel_load_image(path):
    return SpatialDataset.load_single_image(path, id=False)
