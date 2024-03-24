import json
import os

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

from ..configuration.Types import ColName, Splits


class TorchDataset(Dataset):
    """Characterizes a dataset for PyTorch"""

    def __init__(self, img_dir, label_name, transform=None, target_transform=None):
        self.img_dir = img_dir
        self.img_labels = (
            pd.read_csv(os.path.join(self.img_dir, "label.csv"))
            .sample(frac=1)
            .reset_index(drop=True)
        )
        self.label_name = label_name
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        """Denotes the total number of samples"""
        return len(self.img_labels)

    def __getitem__(self, idx):
        """Generates one sample of data"""
        # Get data and label
        label = int(self.img_labels.iloc[idx][self.label_name])
        patch_id = self.img_labels.iloc[idx][ColName.patch_id.value]
        img_path = os.path.join(self.img_dir, f"{label}/patch_{patch_id}.npy")
        image = np.load(img_path)

        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            label = self.target_transform(label)

        return image, label


def make_torch_dataloader(
    data_path: str,
    label_name: str,
    params,
    model_arch,
    normalization_params_path=None,
):

    if normalization_params_path is None:
        normalization_params_path = os.path.join(data_path, "normalization_params.json")

    # Load channel-wise mean and stdev of the training data
    with open(normalization_params_path, "r") as f:
        normalization_params = json.load(f)

    # Define the image transformations
    transformation = [
        transforms.ToTensor(),
        transforms.Normalize(
            normalization_params["mean"],
            normalization_params["stdev"],
        ),
        transforms.ConvertImageDtype(torch.float),
    ]
    if model_arch == "unet":
        train_transform = transforms.Compose(
            transformation
            + [
                transforms.RandomHorizontalFlip(),
                transforms.RandomVerticalFlip(),
                transforms.RandomRotation(degrees=90),
            ]
        )
        test_transform = transforms.Compose(transformation)
    else:
        train_transform = transforms.Compose(
            transformation + [lambda x: torch.mean(x, dim=(1, 2))]
        )
        test_transform = train_transform

    # Define the datasets
    training_data = TorchDataset(
        os.path.join(data_path, Splits.train.value),
        label_name=label_name,
        transform=train_transform,
    )
    validation_data = TorchDataset(
        os.path.join(data_path, Splits.validate.value),
        label_name=label_name,
        transform=test_transform,
    )
    testing_data = TorchDataset(
        os.path.join(data_path, Splits.test.value),
        label_name=label_name,
        transform=test_transform,
    )

    # Define the dataloaders
    train_loader = DataLoader(training_data, shuffle=True, **params)
    val_loader = DataLoader(validation_data, shuffle=False, **params)
    test_loader = DataLoader(testing_data, shuffle=False, **params)

    return train_loader, val_loader, test_loader
