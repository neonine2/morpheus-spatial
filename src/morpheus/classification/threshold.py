import os
import json
from typing import Union, List
import numpy as np
import torch

from .classifier import load_model
from ..datasets.spatial_dataset import SpatialDataset

def optimize_threshold(dataset, split="validate", model_path=None):
    X, y, metadata, model = get_data_and_model(
        dataset,
        model_path=model_path,
        data_split=split,
        remove_small_images=True,
    )
    pred = model(X)
    metadata["pred"] = pred
    metadata["true"] = y

    thresholds = np.linspace(0, 1, 101)
    rmse = []
    for t in thresholds:
        metadata["pred_binary"] = metadata["pred"] > t
        pred = metadata.groupby("ImageNumber").agg(
            {"pred_binary": "mean", "true": "mean"}
        )
        rmse.append(np.sqrt(np.mean((pred["pred_binary"] - pred["true"]) ** 2)))
    return thresholds[np.argmin(rmse)]

def load_classifier(model_path, mu, stdev):
    classifier = load_model(model_path)
    wrapped_classifier = (
        lambda x: classifier(
            torch.permute(torch.from_numpy((x - mu) / stdev).float(), (0, 3, 1, 2))
        )
        .detach()
        .numpy()[:, 1]
    )
    return wrapped_classifier

def load_data_split(
    dataset,
    data_split: Union[str, List[str]],
    remove_small_images=False,
    parallel=False,
):
    # get data
    if isinstance(data_split, list):
        # data_split is a list, use isin to filter
        _metadata = dataset.metadata[dataset.metadata["splits"].isin(data_split)]
    else:
        # data_split is a string, use equality to filter
        _metadata = dataset.metadata[dataset.metadata["splits"] == data_split]

    original_patch_count = _metadata["ImageNumber"].nunique()

    if remove_small_images:
        filter = _metadata.groupby("ImageNumber").count()["Contains_Tumor"] >= 16
        _metadata = _metadata[_metadata["ImageNumber"].isin(filter[filter].index)]

    final_image_count = _metadata["ImageNumber"].nunique()

    X = dataset.load_from_metadata(_metadata, parallel=parallel)
    y = _metadata["Contains_Tcytotoxic"].values.flatten()
    return X, y, _metadata

def get_data_and_model(
    dataset: SpatialDataset,
    data_split: Union[str, List[str]],
    model_path: str = None,
    remove_small_images: bool = False,
    pallalel: bool = False,
):
    # load image data and label
    X, y, metadata = load_data_split(
        dataset,
        data_split,
        remove_small_images,
        parallel=pallalel,
    )

    # Load normalization parameters
    with open(
        os.path.join(dataset.split_dir, "normalization_params.json")
    ) as json_file:
        normalization_params = json.load(json_file)
        mu = normalization_params["mean"]
        stdev = normalization_params["stdev"]

    # load classifier
    model_path = model_path if model_path is not None else dataset.model_path
    model = load_classifier(model_path, mu, stdev)

    return X, y, metadata, model
