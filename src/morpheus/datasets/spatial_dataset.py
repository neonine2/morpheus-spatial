import json
import os
import warnings

import h5py
import numpy as np
import pandas as pd
import ray
from tqdm import tqdm

from ..utils.patchify import generate_patches_optimized
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
        save: bool = True,
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
        if save:
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
        patches_array, metadata_df = generate_patches_optimized(
            df, patch_size, pixel_size, cell_types, self.channel_names, cell_to_mask
        )
        metadata_df = SpatialDataset.convert_object_columns(
            metadata_df.reset_index().rename(columns={"index": ColName.patch_id.value})
        )
        self.metadata = metadata_df
        n, h, w, c = patches_array.shape

        # check number of channel names matches the number of channels in the data
        if c != len(self.channel_names):
            raise ValueError("Number of channel names do not match data dimensions")

        # check the patch dimensions match the expected dimensions
        if h != patch_size or w != patch_size:
            raise ValueError("Patch dimensions do not match the expected dimensions")

        # check the number of patches generated matches the number of rows in the metadata
        if n != len(metadata_df):
            raise ValueError("Number of patches generated does not match metadata")

        if save:
            with h5py.File(self.patch_path, "w") as f:
                # Create a dataset to store the images
                f.create_dataset(
                    "images",
                    data=patches_array,
                    compression="gzip",
                    chunks=(min(n, 100), h, w, c),
                    dtype=patches_array.dtype,
                )

                # Create a dataset to store the channel names
                f.create_dataset("channel_names", data=self.channel_names)

                # Create a dataset to store the metadata
                metadata_numpy = metadata_df.to_records(index=False)
                f.create_dataset(
                    "metadata", data=metadata_numpy, dtype=metadata_numpy.dtype
                )
            print(f"Patches saved to {self.patch_path}")
            self.load_patch_data()
            self.check_loaded_patch()
        print(f"Number of patches generated: {n}")
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
    ):
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

        print("Generating data splits...")
        if specify_split is not None:
            if SpatialDataset.issplitvalid(specify_split):
                patient_split = {
                    name: np.array(spt) for name, spt in specify_split.items()
                }
            else:
                raise ValueError("Given split is not valid")
        else:
            patient_split = self.get_patient_splits(
                stratify_by,
                train_size,
                val_size,
                test_size,
                random_state,
                shuffle,
                **tolerance,
            )

        if patient_split is None:
            raise ValueError(
                "Could not satisfy data split constraints, try again or adjust constraints"
            )

        if save:
            print("Saving splits...")
            self.save_splits(patient_split, label_name=stratify_by)
            print(f"Data splits saved to {self.split_dir}")
            self.get_split_info()
        return patient_split

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

    def save_splits(self, patient_split, label_name=ColName.contains_cd8.value):

        # obtain patch index corresponding to patient split
        split_index = {
            key: self.metadata[self.metadata[ColName.patient_id.value].isin(val)].index
            for key, val in patient_split.items()
        }

        # read in image data
        with h5py.File(self.patch_path, "r") as f:
            patches = f["images"][:]

        # iterate over splits and save patches
        normalization_params = None
        for split_name in tqdm(
            Splits,
            desc="Saving splits",
        ):
            index = split_index[split_name.value]
            _patches = patches[index, ...]
            _labels = self.metadata.iloc[index][label_name].values.astype(int)
            _ids = self.metadata.iloc[index][ColName.patch_id.value].values
            metadata_to_save = self.metadata.iloc[index][
                [
                    ColName.patch_id.value,
                    label_name,
                    ColName.patient_id.value,
                    ColName.image_id.value,
                ]
            ]

            # make directories for the split
            _path = os.path.join(self.split_dir, split_name.value)
            if not os.path.isdir(_path):
                os.makedirs(_path)
                os.makedirs(os.path.join(_path, "0"))
                os.makedirs(os.path.join(_path, "1"))

            # save metadata
            metadata_to_save.to_csv(
                os.path.join(_path, DefaultFileName.label.value), index=False
            )

            # save patches
            n_image = len(_labels)
            for i in tqdm(
                range(n_image),
                desc=f"Saving images for {split_name.value} split",
                leave=False,
            ):
                # sparse_tensor = torch.tensor(_patches[i, ...]).to_sparse()
                save_path = os.path.join(_path, f"{_labels[i]}/patch_{_ids[i]}.npy")
                # Save the sparse tensor
                np.save(save_path, _patches[i, ...])
                # torch.save(sparse_tensor, save_path)

            # save normalization parameters
            if split_name.value == Splits.train.value:
                normalization_params = {
                    "mean": np.mean(_patches, axis=(0, 1, 2)).tolist(),
                    "stdev": np.std(_patches, axis=(0, 1, 2)).tolist(),
                }

        # save normalization parameters to json
        with open(os.path.join(self.split_dir, "normalization_params.json"), "w") as f:
            json.dump(normalization_params, f)

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
