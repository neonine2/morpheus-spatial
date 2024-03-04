import os
import pickle
import pandas as pd
import numpy as np
import torch
from torchvision import transforms
from torch.utils.data import DataLoader, Dataset


def set_seed(seed):
    np.random.seed(seed)


def set_pytorch_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():  # GPU operation have separate seed
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)


class SpatialDataset(Dataset):
    "Characterizes a dataset for PyTorch"

    def __init__(self, img_dir, transform=None, target_transform=None):
        self.img_dir = img_dir
        self.img_labels = pd.read_csv(os.path.join(self.img_dir, "label.csv"))
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        "Denotes the total number of samples"
        return len(self.img_labels)

    def __getitem__(self, idx):
        "Generates one sample of data"
        # Get data and label
        label = self.img_labels.iloc[idx, 1]
        img_path = os.path.join(self.img_dir, f"{label}/patch_{idx}.npy")
        image = np.load(img_path)
        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            label = self.target_transform(label)
        return image, label


def make_torch_dataloader(
    data_path,
    model="unet",
    params={"batch_size": 64, "num_workers": 4, "pin_memory": True},
):

    # Load the data info which should contain the channel-wise mean and stdev of the training data
    with open(os.path.join(data_path, "split_info.pkl"), "rb") as f:
        info_dict = pickle.load(f)

    # Define the image transformations
    transformation = [
        transforms.ToTensor(),
        transforms.Normalize(info_dict["train_set_mean"], info_dict["train_set_stdev"]),
        transforms.ConvertImageDtype(torch.float),
    ]
    if model == "mlp" or model == "lr":
        train_transform = transforms.Compose(
            transformation + [lambda x: torch.mean(x, dim=(1, 2))]
        )
        test_transform = train_transform
    else:
        train_transform = transforms.Compose(
            transformation
            + [
                transforms.RandomHorizontalFlip(),
                transforms.RandomVerticalFlip(),
                transforms.RandomRotation(degrees=90),
            ]
        )
        test_transform = transforms.Compose(transformation)

    # Define the datasets
    training_data = SpatialDataset(
        os.path.join(data_path, "train"), transform=train_transform
    )
    validation_data = SpatialDataset(
        os.path.join(data_path, "validate"), transform=test_transform
    )
    testing_data = SpatialDataset(
        os.path.join(data_path, "test"), transform=test_transform
    )

    # Define the dataloaders
    train_loader = DataLoader(training_data, shuffle=True, **params)
    val_loader = DataLoader(validation_data, shuffle=False, **params)
    test_loader = DataLoader(testing_data, shuffle=False, **params)

    return train_loader, val_loader, test_loader
