import json
import os

import h5py
import numpy as np
import pandas as pd

from ..configuration.Types import CellType, ColName, Splits


class SpatialDataset:
    def __init__(self, data_path: str):
        self.data_dim = None
        self.channel_names = None
        self.metadata = None
        self.data_path = data_path
        self.load_data()
        self.check_data()

    def check_data(self):
        # check that the data is loaded
        if not hasattr(self, "metadata"):
            raise ValueError("Metadata not loaded")
        if not hasattr(self, "channel_names"):
            raise ValueError("Channel names not loaded")

        # check that the data is consistent
        if len(self.metadata) != self.metadata[ColName.patch_id.value].nunique():
            raise ValueError("Metadata contains duplicate patch IDs")
        if len(self.channel_names) != self.data_dim[-1]:
            raise ValueError("Number of channel names do not match data dimensions")

        # check key metadata columns are present
        for col in [ColName.patient_id.value, ColName.patch_id.value]:
            if col not in self.metadata.columns:
                raise ValueError(f"Metadata missing column: {col}")

    def load_data(self):
        try:
            with h5py.File(self.data_path, "r") as f:
                self.metadata = pd.DataFrame(f["metadata"][:])
                self.channel_names = [
                    name.decode("utf-8") for name in f["channel_names"][:]
                ]
                self.data_dim = f["images"].shape
        except Exception as e:
            print(f"Error loading data: {e}")

    def generate_data_splits(
        self,
        stratify_by: str,
        train_size=None,
        val_size=None,
        test_size=None,
        save_dir=None,
        random_state=None,
        shuffle=True,
        tolerance=None,
        given_splits=None,
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
        save: bool
            Whether to save the data splits to the save directory
        """
        if tolerance is None:
            tolerance = {"eps": 0.01, "train_lb": 0.5, "n_tol": 100}
        if save_dir is None:
            self.save_dir = os.path.join(
                os.path.dirname(self.data_path), "data"
            )  # default save directory
        else:
            self.save_dir = save_dir

        if os.path.isdir(os.path.join(self.save_dir, Splits.train.value)):
            print(f"Data splits already exist in {self.save_dir}")
            return

        print("Generating data splits...")
        if given_splits is not None:
            patient_split = {
                val: np.array(given_splits[idx])
                for idx, val in enumerate(
                    [Splits.train.value, Splits.validate.value, Splits.test.value]
                )
            }
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
            print(
                "Could not satisfy data split constraints, try again or adjust constraints"
            )
            return

        if save:
            print("Saving splits...")
            self.save_splits(patient_split, label_name=stratify_by)
            print(f"Data splits saved to {self.save_dir}")
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
        train_size=0.6,
        val_size=0.2,
        test_size=0.2,
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

    def save_splits(self, patient_split, label_name=CellType.cd8.value):
        # import torch
        from tqdm import tqdm

        # obtain patch index corresponding to patient split
        split_index = {
            key: self.metadata[self.metadata[ColName.patient_id.value].isin(val)].index
            for key, val in patient_split.items()
        }

        # read in image data
        with h5py.File(self.data_path, "r") as f:
            patches = f["images"][:]

        # iterate over splits and save patches
        normalization_params = None
        for split_name in tqdm(
            [Splits.train.value, Splits.validate.value, Splits.test.value],
            desc="Saving splits",
        ):
            index = split_index[split_name]
            _patches = patches[index, ...]
            _labels = self.metadata.iloc[index][label_name].values
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
            _path = os.path.join(self.save_dir, split_name)
            if not os.path.isdir(_path):
                os.makedirs(_path)
                os.makedirs(os.path.join(_path, "0"))
                os.makedirs(os.path.join(_path, "1"))

            # save metadata
            metadata_to_save.to_csv(os.path.join(_path, "label.csv"), index=False)

            # save patches
            n_image = len(_labels)
            for i in tqdm(
                range(n_image),
                desc=f"Saving images for {split_name} split",
                leave=False,
            ):
                # sparse_tensor = torch.tensor(_patches[i, ...]).to_sparse()
                save_path = os.path.join(_path, f"{_labels[i]}/patch_{_ids[i]}.npy")
                # Save the sparse tensor
                np.save(save_path, _patches[i, ...])
                # torch.save(sparse_tensor, save_path)

            # save normalization parameters
            if split_name == Splits.train.value:
                normalization_params = {
                    "mean": np.mean(_patches, axis=(0, 1, 2)).tolist(),
                    "stdev": np.std(_patches, axis=(0, 1, 2)).tolist(),
                }

        # save normalization parameters to json
        with open(os.path.join(self.save_dir, "normalization_params.json"), "w") as f:
            json.dump(normalization_params, f)
