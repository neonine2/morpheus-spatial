import os
import json
import _pickle as pickle
import numpy as np
import pandas as pd
from typing import Optional

import torch

import ray
from tqdm import tqdm

from ..datasets import SpatialDataset
from .cf import Counterfactual
from ..classification import load_model, optimize_threshold_chunked
from ..confidence import TrustScore
from ..configuration import (
    Splits,
    ColName,
    DefaultFolderName,
    DefaultFileName,
)

EPSILON = torch.tensor(1e-20, dtype=torch.float32)


def get_counterfactual(
    images: pd.DataFrame,
    dataset: SpatialDataset,
    target_class: int,
    model_path: str,
    optimization_params: dict,
    kdtree_path: str = None,
    save_dir: str = None,
    num_workers: int = 1,
    train_data: str = None,
    verbosity: int = 0,
    trustscore_kwargs: Optional[dict] = None,
    device: str = None,
    model_kwargs: Optional[dict] = {},
    tumor_name: str = "Tumor",
    cd8_name: str = "Tcytotoxic",
):
    """
    Generate counterfactuals for the dataset.

    Args:
        dataset (SpatialDataset): Dataset to generate counterfactuals for.
        target_class (int): Target class for the counterfactuals.
        model (torch.nn.Module): Model to generate counterfactuals for.
        optimization_params (dict): Dictionary containing the parameters for the optimization.
        images (pd.DataFrame, optional): Images to generate counterfactuals for. Defaults to None.
        kdtree_path (str, optional): Path to the kdtree file. Defaults to None.
        save_dir (str, optional): Directory where output will be saved. Defaults to None.
        num_workers (bool, optional): Number of workers to use for parallel processing. Defaults to None.
        train_data (str, optional): Path to the training data. Defaults to None.
        verbosity (int, optional): Verbosity level. Defaults to 0.
        trustscore_kwargs (dict, optional): Dictionary containing the parameters for the trustscore. Defaults to None.
        device (str, optional): Device to use for computation. Defaults to None.
        model_kwargs (dict, optional): Additional keyword arguments for the model. Defaults to {}.
    """

    # Get optimal model threshold by maximizing RMSE over validation set
    print("Optimizing threshold using validation set...")
    opt_cutoff = optimize_threshold_chunked(
        dataset,
        split="validate",
        model_path=model_path,
        tumor_name=tumor_name,
        cd8_name=cd8_name,
    )
    print(f"Optimal threshold: {opt_cutoff}")
    # Set default values
    threshold = optimization_params.pop("threshold", opt_cutoff)
    channel_to_perturb = optimization_params.pop("channel_to_perturb", None)

    # set default tensor type to cuda if available
    torch.set_default_tensor_type(
        torch.FloatTensor if torch.cuda.is_available() else torch.FloatTensor
    )

    # Load normalization parameters
    with open(
        os.path.join(dataset.split_dir, DefaultFileName.normalization.value)
    ) as json_file:
        normalization_params = json.load(json_file)
        mu = normalization_params["mean"]
        stdev = normalization_params["stdev"]

    # Set default paths
    if train_data is None:
        train_data = os.path.join(dataset.split_dir, Splits.train.value)
    if kdtree_path is None:
        kdtree_path = os.path.join(dataset.root_dir, DefaultFileName.kdtree.value)
    if images is None:
        images = dataset.metadata

    # Create save directory
    if save_dir is None:
        save_dir = os.path.join(
            dataset.root_dir, DefaultFolderName.counterfactual.value
        )
    os.makedirs(save_dir, exist_ok=True)

    # Build kdtree if it does not exist
    model = load_model(model_path, **model_kwargs).to(device)
    if not os.path.exists(kdtree_path):
        print("Building kdtree from data...")
        build_kdtree(kdtree_path, train_data, model, mu, stdev, trustscore_kwargs)
        print("kdtree saved")

    # Organize arguments for counterfactual generation

    # used across all images
    general_args = {
        "target_class": target_class,
        "model_path": model_path,
        "channel": dataset.channel_names,
        "channel_to_perturb": channel_to_perturb,
        "mu": mu,
        "stdev": stdev,
        "kdtree_path": kdtree_path,
        "verbosity": verbosity,
        "optimization_params": optimization_params,
        "save_dir": save_dir,
        "device": device,
        "model_kwargs": model_kwargs,
    }

    # specific to each image
    image_args = []
    for i in range(len(images)):
        img_path = dataset.generate_patch_path(
            patch_id=images.iloc[i][ColName.patch_id.value],
            label=images.iloc[i][dataset.label_name],
            split=images.iloc[i][ColName.splits.value],
        )
        img, patch_id = SpatialDataset.load_single_image(img_path)
        label = dataset.metadata.iloc[patch_id][dataset.label_name]
        image_args.append(
            {
                "original_patch": img,
                "original_class": label,
                "patch_id": patch_id,
            }
        )

    print(
        "Applying model to all instances to filter out ones already classified as target class"
    )
    discard_mask = [
        args["original_class"] == target_class
        or model(
            torch.from_numpy(
                np.transpose((args["original_patch"] - mu) / stdev, (2, 0, 1))[None, :]
            )
            .float()
            .to(device)
        )
        .detach()
        .cpu()
        .numpy()[0, 1]
        > threshold
        for args in tqdm(image_args, miniters=100)
    ]
    image_args = [args for i, args in enumerate(image_args) if not discard_mask[i]]

    # Save hyperparameters
    hyper_path = os.path.join(save_dir, "hyperparameters.json")
    with open(hyper_path, "w") as json_file:
        json.dump(
            {
                "target_class": target_class,
                "channel_to_perturb": channel_to_perturb,
                "optimization_params": optimization_params,
                "threshold": threshold,
            },
            json_file,
        )
    print(f"hyperparameters saved to {hyper_path}")

    # Generate counterfactuals
    print("Generating counterfactuals...")
    if num_workers > 1:
        # Initialize Ray
        ray.shutdown()
        ray.init()

        # Get the number of available CPUs
        num_cpus = ray.available_resources()["CPU"]
        print(f"Number of available CPUs: {num_cpus}")

        # Create a list of Ray object references
        cf_refs = [
            generate_one_cf_wrapper.remote({**args, **general_args})
            for args in image_args
        ]

        # Use tqdm to display a progress bar
        with tqdm(total=len(image_args), miniters=100) as pbar:
            # Retrieve the results as they become available
            results = []
            while cf_refs:
                done_refs, cf_refs = ray.wait(cf_refs)
                results.extend(ray.get(done_refs))
                pbar.update(len(done_refs))

        # Shutdown Ray
        ray.shutdown()
    else:
        for args in tqdm(image_args, total=len(image_args)):
            generate_one_cf(**{**args, **general_args})
    print("Countefactual generation completed!")


@ray.remote
def generate_one_cf_wrapper(combined_args):
    torch.set_num_threads(1)  # very important
    return generate_one_cf(**combined_args)


def generate_one_cf(
    original_patch: np.ndarray,
    original_class: np.ndarray,
    patch_id: int,
    target_class: int,
    model_path: str,
    channel: list,
    channel_to_perturb: list,
    mu: np.ndarray,
    stdev: np.ndarray,
    kdtree_path: str,
    verbosity: int = 0,
    optimization_params: dict = None,
    save_dir: str = None,
    device: str = None,
    model_kwargs: Optional[dict] = {},
) -> None:
    """
    Generate counterfactuals for a given image patch.

    Args:
         original_patch (np.ndarray): Original patch to be explained.
         original_class (np.ndarray): Original label of the patch.
            target_class (int): Target class for the counterfactual.
            model_path (str): Path to the model.
         channel_to_perturb (list): List of channels to perturb.
         normalization_params (dict): Dictionary containing the mean and standard deviation of each channel.
         train_data (str, optional): Path to the training data. Defaults to None.
         optimization_params (dict, optional): Dictionary containing the parameters for the optimization. Defaults to {}.
         save_dir (str, optional): Directory where output will be saved. Defaults to None.
         patch_id (int, optional): Patch ID. Defaults to None.
         device (str, optional): Device to use for computation. Defaults to None.
         model_kwargs (dict, optional): Additional keyword arguments for the model. Defaults to {}.

     Returns:
         None
    """
    # load model
    model = load_model(model_path, **model_kwargs).to(device)

    # Obtain data features
    stdev, mu = (
        torch.tensor(stdev).float(),
        torch.tensor(mu).float(),
    )
    H, _, C = original_patch.shape
    original_patch = torch.from_numpy(original_patch.copy()).float().to(device)
    original_patch = (original_patch - mu) / stdev
    original_class = torch.tensor([original_class], dtype=torch.int64).to(device)
    X_mean = torch.mean(original_patch, dim=(0, 1)).to(device)

    # Adding init layer to model
    unnormed_mean = X_mean * stdev + mu
    unnormed_patch = original_patch[None, :] * stdev + mu
    init_fun = lambda y: alter_image(y, unnormed_patch, mu, stdev, unnormed_mean)
    altered_model, input_transform = add_init_layer(init_fun, model)

    # Set range of each channel to perturb
    is_perturbed = np.array(
        [True if name in channel_to_perturb else False for name in channel]
    )
    feature_range = (
        torch.maximum(-mu / stdev, torch.ones(C) * -4),
        torch.ones(C) * 4,
    )
    feature_range[0][~is_perturbed] = X_mean[~is_perturbed] - EPSILON
    feature_range[1][~is_perturbed] = X_mean[~is_perturbed] + EPSILON

    # define predict function
    predict_fn = lambda x: altered_model(x)

    # define counterfactual object
    shape = (1,) + X_mean.shape
    cf = Counterfactual(
        predict_fn,
        input_transform,
        shape,
        feature_range=feature_range,
        trustscore=kdtree_path,
        verbosity=verbosity,
        device=device,
        **optimization_params,
    )
    cf.fit()

    # generate counterfactual
    explanation = cf.explain(
        X=X_mean[None, :], Y=original_class[None, :], target_class=[target_class]
    )

    if explanation.cf is not None:
        cf_prob = explanation.cf["proba"][0]
        if verbosity > 0:
            print(f"Counterfactual probability: {cf_prob}")

        cf = explanation.cf["X"][0]
        cf = input_transform(torch.from_numpy(cf[None, :]).to(device))
        cf = torch.permute(cf, (0, 2, 3, 1))

        X_perturbed = mean_preserve_dimensions(
            cf * stdev + mu, preserveAxis=cf.ndim - 1
        )
        original_patch = X_mean * stdev + mu
        cf_delta = (X_perturbed - original_patch) / original_patch * 100
        percent_delta = dict(
            zip(
                np.array(channel)[is_perturbed],
                cf_delta[is_perturbed].cpu().numpy(),
            )
        )
        if verbosity > 0:
            print(f"cf perturbed (%): {percent_delta}")

        if save_dir is not None:
            os.makedirs(save_dir, exist_ok=True)
            saved_file = os.path.join(save_dir, f"patch_{patch_id}.npz")
            np.savez(
                saved_file,
                cf=X_perturbed,
                proba=cf_prob,
                delta_in_percentage=percent_delta,
            )
        return percent_delta


def build_kdtree(
    kdtree_path,
    train_data: str,
    model: torch.nn.Module,
    mu: list,
    stdev: list,
    trustscore_kwargs: Optional[dict] = None,
    batch_size: int = 512  # Batch size to control memory usage
):
    if trustscore_kwargs is not None:
        ts = TrustScore(**trustscore_kwargs)
    else:
        ts = TrustScore()

    # Variables to accumulate the mean calculation
    chunk_means = []
    all_preds = []

    # Process the data incrementally in batches
    print("processing data in batches")
    for batch in tqdm(load_npy_files_in_batches(train_data, batch_size=batch_size)):
        # Normalize and permute the batch
        batch = torch.from_numpy((batch - np.array(mu)) / np.array(stdev)).float()
        X_t = torch.permute(batch, (0, 3, 1, 2))

        # Pass through the model to get predictions
        model_out = model(X_t).detach().numpy()
        preds = np.argmax(model_out, axis=1)
        all_preds.append(preds)

        # Compute mean across the spatial dimensions for the batch
        chunk_mean = torch.mean(batch, dim=(1, 2))
        chunk_means.append(chunk_mean)

    # Fit TrustScore using the overall mean
    overall_mean = torch.cat(chunk_means, dim=0)
    preds_flattened = np.concatenate(all_preds)
    ts.fit(overall_mean, preds_flattened, classes=2)
    save_object(ts, kdtree_path)


def load_npy_files_in_batches(base_dir, batch_size=128):
    """Generator that loads the .npy files in batches to reduce memory usage."""
    arrays = [] # List to store the arrays
    sub_dirs = ["0", "1"]  # Subdirectories to look into

    for sub_dir in sub_dirs:
        sub_dir_path = os.path.join(base_dir, sub_dir)
        for file in os.listdir(sub_dir_path):
            if file.endswith(".npy"):
                file_path = os.path.join(sub_dir_path, file)
                array = np.load(file_path)
                arrays.append(array)

            # Once we reach the batch size, yield the batch
            if len(arrays) >= batch_size:
                yield np.stack(arrays, axis=0)
                arrays = []  # Reset for the next batch

    # Yield any remaining arrays
    if arrays:
        yield np.stack(arrays, axis=0)


def alter_image(y, unnormed_patch, mu, stdev, unnormed_mean):
    unnormed_y = y * stdev + mu
    new_patch = unnormed_patch * ((unnormed_y / unnormed_mean)[:, None, None, :])
    return (new_patch - mu) / stdev


def add_init_layer(init_fun, model):
    """
    Add an initialization layer to the model.

    Args:
        init_fun (callable): Initialization function.
        model (torch.nn.Module): Original model.

    Returns:
        tuple: (torch.nn.Module, torch.nn.Module) - Modified model and input transformation layer.
    """

    class InputFun(torch.nn.Module):
        def forward(self, input):
            return torch.permute(init_fun(input), (0, 3, 1, 2)).float()

    input_transform = InputFun()
    complete_model = torch.nn.Sequential(input_transform, model)
    return complete_model, input_transform


def mean_preserve_dimensions(
    tensor: torch.Tensor, preserveAxis: tuple = None
) -> torch.Tensor:
    """
    Compute the mean along all dimensions except those specified in preserveAxis.

    Args:
        tensor (torch.Tensor): Input tensor.
        preserveAxis (tuple, optional): Dimensions to preserve. Defaults to None.

    Returns:
        torch.Tensor: Tensor with preserved dimensions.
    """
    if isinstance(preserveAxis, int):
        preserveAxis = (preserveAxis,)

    dims_to_reduce = [i for i in range(tensor.ndim) if i not in preserveAxis]
    result = tensor.mean(dim=dims_to_reduce)
    return result


def save_object(obj, filename):
    with open(filename, "wb") as outp:  # Overwrites any existing file.
        pickle.dump(obj, outp, -1)
