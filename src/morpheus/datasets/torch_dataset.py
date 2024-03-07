import os
import json
import pandas as pd
import numpy as np
import torch
from torchvision import transforms
from torch.utils.data import DataLoader, Dataset
from ..constants import splits, colname


class TorchDataset(Dataset):
    "Characterizes a dataset for PyTorch"

    def __init__(self, img_dir, labelname, transform=None, target_transform=None):
        self.img_dir = img_dir
        self.img_labels = pd.read_csv(os.path.join(self.img_dir, "label.csv"))
        self.labelname = labelname
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        "Denotes the total number of samples"
        return len(self.img_labels)

    def __getitem__(self, idx):
        "Generates one sample of data"
        # Get data and label
        label = self.img_labels.iloc[idx][self.labelname]
        id = self.img_labels.iloc[idx][colname.patch_id.value]
        img_path = os.path.join(self.img_dir, f"{label}/patch_{id}.npy")
        image = np.load(img_path)

        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            label = self.target_transform(label)
        return image, label


def make_torch_dataloader(
    data_path: str,
    labelname: str,
    model_arch="unet",
    normalization_params_path=None,
    params={"batch_size": 64, "num_workers": 4, "pin_memory": True},
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
            normalization_params["mean"], normalization_params["stdev"]
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
        os.path.join(data_path, splits.train.value),
        labelname=labelname,
        transform=train_transform,
    )
    validation_data = TorchDataset(
        os.path.join(data_path, splits.validate.value),
        labelname=labelname,
        transform=test_transform,
    )
    testing_data = TorchDataset(
        os.path.join(data_path, splits.test.value),
        labelname=labelname,
        transform=test_transform,
    )

    # Define the dataloaders
    train_loader = DataLoader(training_data, shuffle=True, **params)
    val_loader = DataLoader(validation_data, shuffle=False, **params)
    test_loader = DataLoader(testing_data, shuffle=False, **params)

    return train_loader, val_loader, test_loader
