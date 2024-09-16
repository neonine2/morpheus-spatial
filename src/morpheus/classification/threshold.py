import os
import json
from typing import Union, List
import numpy as np
import torch

from .classifier import load_model
from ..datasets.spatial_dataset import SpatialDataset


def optimize_threshold(
    dataset,
    split="validate",
    model_path=None,
    tumor_name="Tumor",
    cd8_name="Tcytotoxic",
):
    X, y, metadata, model = get_data_and_model(
        dataset,
        model_path=model_path,
        data_split=split,
        remove_small_images=True,
        label_col=f"Contains_{cd8_name}",
        tumor_col=f"Contains_{tumor_name}",
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

def chunkwise_prediction(metadata, label_col, dataset, model, chunk_size=100):
    all_preds = []
    num_rows = metadata.shape[0]
    for start in range(0, num_rows, chunk_size):
        metadata_chunk = metadata.iloc[start:(start + chunk_size)]
        X_chunk = dataset.load_from_metadata(metadata_chunk, col_as_label=label_col, parallel=False)
        pred_chunk = model(X_chunk)
        all_preds.append(pred_chunk)
    final_pred = np.concatenate(all_preds, axis=0)

    return final_pred

def optimize_threshold_chunked(
    dataset,
    split="validate",
    model_path=None,
    tumor_name="Tumor",
    cd8_name="Tcytotoxic",
):
    metadata, model = get_metadata_and_model(
        dataset,
        model_path=model_path,
        data_split=split,
        remove_small_images=True,
        tumor_col=f"Contains_{tumor_name}",
    )

    label_col = f"Contains_{cd8_name}"
    metadata["pred"] = chunkwise_prediction(metadata, label_col, dataset, model)
    metadata["true"] = metadata[label_col].values.flatten()

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

def load_metadata_split(
    dataset,
    data_split: Union[str, List[str]],
    tumor_col: str,
    remove_small_images=False,
):
    # get data
    if isinstance(data_split, list):
        # data_split is a list, use isin to filter
        _metadata = dataset.metadata[dataset.metadata["splits"].isin(data_split)]
    else:
        # data_split is a string, use equality to filter
        _metadata = dataset.metadata[dataset.metadata["splits"] == data_split]

    if remove_small_images:
        filter = _metadata.groupby("ImageNumber").count()[tumor_col] >= 16
        _metadata = _metadata[_metadata["ImageNumber"].isin(filter[filter].index)]

    return _metadata

def get_metadata_and_model(
    dataset: SpatialDataset,
    data_split: Union[str, List[str]],
    model_path: str = None,
    remove_small_images: bool = False,
    tumor_col: str = "Contains_Tumor",
):
    # load image data and label
    metadata = load_metadata_split(
        dataset=dataset,
        data_split=data_split,
        remove_small_images=remove_small_images,
        tumor_col=tumor_col
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

    return metadata, model

def load_data_split(
    dataset,
    data_split: Union[str, List[str]],
    label_col: str,
    tumor_col: str,
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

    if remove_small_images:
        filter = _metadata.groupby("ImageNumber").count()[tumor_col] >= 16
        _metadata = _metadata[_metadata["ImageNumber"].isin(filter[filter].index)]

    X = dataset.load_from_metadata(_metadata, col_as_label=label_col, parallel=parallel)
    y = _metadata[label_col].values.flatten()
    return X, y, _metadata


def get_data_and_model(
    dataset: SpatialDataset,
    data_split: Union[str, List[str]],
    model_path: str = None,
    remove_small_images: bool = False,
    pallalel: bool = False,
    label_col: str = "Contains_Tcytotoxic",
    tumor_col: str = "Contains_Tumor",
):
    # load image data and label
    X, y, metadata = load_data_split(
        dataset=dataset,
        data_split=data_split,
        remove_small_images=remove_small_images,
        tumor_col=tumor_col,
        label_col=label_col,
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
