import os
import json
import _pickle as pickle
import numpy as np
import pandas as pd
from typing import Optional
import torch

from tqdm import tqdm

# import multiprocessing


from ..datasets import SpatialDataset
from .cf import Counterfactual
from ..confidence import TrustScore
from ..configuration import (
    Splits,
    ColName,
    DefaultFolderName,
    DefaultFileName,
)

EPSILON = torch.tensor(1e-20, dtype=torch.float32)


def get_counterfactual(
    dataset: SpatialDataset,
    target_class: int,
    model: torch.nn.Module,
    channel_to_perturb: list,
    optimization_params: dict,
    images: pd.DataFrame = None,
    threshold: float = 0.5,
    kdtree_path: str = None,
    save_dir: str = None,
    num_workers: int = 1,
    train_data: str = None,
    verbose: bool = False,
    trustscore_kwargs: Optional[dict] = None,
):
    """
    Generate counterfactuals for the dataset.

    Args:
        dataset (SpatialDataset): Dataset to generate counterfactuals for.
        target_class (int): Target class for the counterfactuals.
        model (torch.nn.Module): Model to generate counterfactuals for.
        channel_to_perturb (list): List of channels to perturb.
        optimization_params (dict): Dictionary containing the parameters for the optimization.
        images (pd.DataFrame, optional): Images to generate counterfactuals for. Defaults to None.
        threshold (float, optional): Threshold for the prediction probability. Defaults to 0.5.
        kdtree_path (str, optional): Path to the kdtree file. Defaults to None.
        save_dir (str, optional): Directory where output will be saved. Defaults to None.
        num_workers (bool, optional): Number of workers to use for parallel processing. Defaults to None.
        train_data (str, optional): Path to the training data. Defaults to None.
    """
    num_images = len(images)

    # Load normalization parameters
    with open(
        os.path.join(dataset.split_dir, "normalization_params.json")
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
    dataset.cf_dir = save_dir

    # Build kdtree if it does not exist
    if not os.path.exists(kdtree_path):
        build_kdtree(kdtree_path, train_data, model, mu, stdev, trustscore_kwargs)
        print("kdtree saved")

    # Prepare arguments for generate_one_cf function
    generate_one_cf_args = []
    for i in range(num_images):
        img_path = dataset.generate_patch_path(
            patch_id=images.iloc[i][ColName.patch_id.value],
            label=images.iloc[i][dataset.label_name],
            split=images.iloc[i][ColName.splits.value],
        )
        img, patch_id = SpatialDataset.load_single_image(img_path)
        label = dataset.metadata.iloc[patch_id][dataset.label_name]
        generate_one_cf_args.append(
            (
                img,
                label,
                target_class,
                model,
                dataset.channel_names,
                channel_to_perturb,
                mu,
                stdev,
                kdtree_path,
                verbose,
                optimization_params,
                save_dir,
                patch_id,
                threshold,
            )
        )

    # Generate counterfactuals
    parallel = True if num_workers > 1 else False
    if parallel:
        from dask import delayed
        import dask.multiprocessing

        @delayed
        def generate_one_cf_delayed(args):
            return generate_one_cf(*args)

        dask_tasks = [generate_one_cf_delayed(args) for args in generate_one_cf_args]
        _ = dask.compute(*dask_tasks, scheduler="processes", num_workers=num_workers)
    else:
        for args in tqdm(generate_one_cf_args, total=num_images):
            generate_one_cf(*args)


def generate_one_cf(
    original_patch: np.ndarray,
    original_class: np.ndarray,
    target_class: int,
    model: torch.nn.Module,
    channel: list,
    channel_to_perturb: list,
    mu: np.ndarray,
    stdev: np.ndarray,
    kdtree_path: str,
    verbose: bool = False,
    optimization_params: dict = None,
    save_dir: str = None,
    patch_id: int = None,
    threshold: float = 0.5,
) -> None:
    """
    Generate counterfactuals for a given image patch.

    Args:
         original_patch (np.ndarray): Original patch to be explained.
         original_class (np.ndarray): Original label of the patch.
         model (torch.nn.Module): Model to be explained.
         channel_to_perturb (list): List of channels to perturb.
         normalization_params (dict): Dictionary containing the mean and standard deviation of each channel.
         train_data (str, optional): Path to the training data. Defaults to None.
         optimization_params (dict, optional): Dictionary containing the parameters for the optimization. Defaults to {}.
         save_dir (str, optional): Directory where output will be saved. Defaults to None.
         patch_id (int, optional): Patch ID. Defaults to None.
         threshold (float, optional): Threshold for the prediction probability. Defaults to 0.5.

     Returns:
         None
    """
    # Obtain data features
    stdev, mu = (
        torch.tensor(stdev).float(),
        torch.tensor(mu).float(),
    )
    H, _, C = original_patch.shape
    original_patch = (torch.from_numpy(original_patch).float() - mu) / stdev
    original_class = torch.tensor([original_class], dtype=torch.int64)
    X_mean = torch.mean(original_patch, dim=(0, 1))

    if model.arch == "mlp":
        original_patch = X_mean

    # Adding init layer to model
    unnormed_mean = X_mean * stdev + mu
    if model.arch == "mlp":
        altered_model = lambda x: torch.nn.functional.softmax(model(x), dim=1)
        input_transform = lambda x: x
    else:
        unnormed_patch = original_patch[None, :] * stdev + mu
        init_fun = lambda y: alter_image(y, unnormed_patch, mu, stdev, unnormed_mean)
        altered_model, input_transform = add_init_layer(init_fun, model)

    # Set range of each channel to perturb
    is_perturbed = np.array(
        [True if name in channel_to_perturb else False for name in channel]
    )
    feature_range = (torch.maximum(-mu / stdev, torch.ones(C) * -4), torch.ones(C) * 4)
    feature_range[0][~is_perturbed] = X_mean[~is_perturbed] - EPSILON
    feature_range[1][~is_perturbed] = X_mean[~is_perturbed] + EPSILON

    # define predict function
    predict_fn = lambda x: altered_model(x)

    # Terminate if model incorrectly classifies patch as the target class
    orig_proba = predict_fn(X_mean[None, :]).detach().cpu().numpy()
    if verbose:
        print(f"Initial probability: {orig_proba}")
    pred = orig_proba[0, 1] > threshold
    if pred == target_class:
        print("Instance already classified as target class, no counterfactual needed")
        return

    # define counterfactual object
    shape = (1,) + original_patch.shape
    cf = Counterfactual(
        predict_fn,
        input_transform,
        shape,
        feature_range=feature_range,
        trustscore=kdtree_path,
        verbose=verbose,
        **optimization_params,
    )
    cf.fit()

    # generate counterfactual
    explanation = cf.explain(
        X=X_mean[None, :], Y=original_class[None, :], target_class=[target_class]
    )

    if explanation.cf is not None:
        cf_prob = explanation.cf["proba"][0]
        cf = explanation.cf["X"][0]

        # manually compute probability of cf
        cf = input_transform(torch.from_numpy(cf[None, :]))
        counterfactual_probabilities = (
            altered_model(cf) if model.arch == "mlp" else model(cf)
        )
        if model.arch != "mlp":
            cf = torch.permute(cf, (0, 2, 3, 1))

        print(f"Counterfactual probability: {counterfactual_probabilities}")
        X_perturbed = mean_preserve_dimensions(
            cf * stdev + mu, preserveAxis=cf.ndim - 1
        )
        original_patch = X_mean * stdev + mu
        cf_delta = (X_perturbed - original_patch) / original_patch * 100
        cf_perturbed = dict(
            zip(
                np.array(channel)[is_perturbed],
                cf_delta[is_perturbed].numpy(),
            )
        )
        print(f"cf perturbed: {cf_perturbed}")

        if save_dir is not None:
            os.makedirs(save_dir, exist_ok=True)
            saved_file = os.path.join(save_dir, f"patch_{patch_id}.npz")
            np.savez(
                saved_file,
                explanation=explanation,
                cf_perturbed=cf_perturbed,
                channel_to_perturb=np.array(channel)[is_perturbed],
            )
    return explanation


def build_kdtree(
    kdtree_path,
    train_data: str,
    model: torch.nn.Module,
    mu: list,
    stdev: list,
    trustscore_kwargs: Optional[dict] = None,
):
    train_patch = load_npy_files_to_array(train_data)
    train_patch = torch.from_numpy(
        (train_patch - np.array(mu)) / np.array(stdev)
    ).float()
    if model.arch == "mlp":
        X_t = torch.mean(train_patch, (1, 2))
    else:
        X_t = torch.permute(train_patch, (0, 3, 1, 2))

    model_out = model(X_t).detach().numpy()
    preds = np.argmax(model_out, axis=1)

    if trustscore_kwargs is not None:
        ts = TrustScore(**trustscore_kwargs)
    else:
        ts = TrustScore()
    train_patch = torch.mean(train_patch, (1, 2))
    ts.fit(train_patch, preds, classes=2)
    save_object(ts, kdtree_path)


def load_npy_files_to_array(base_dir):
    arrays = []  # This list will hold all the numpy arrays
    sub_dirs = ["0", "1"]  # Subdirectories to look into

    for sub_dir in sub_dirs:
        sub_dir_path = os.path.join(base_dir, sub_dir)
        for file in os.listdir(sub_dir_path):
            if file.endswith(".npy"):
                file_path = os.path.join(sub_dir_path, file)
                array = np.load(file_path)
                arrays.append(array)

    # Stack the arrays to form a single n by l by w by n_channels array
    final_array = np.stack(arrays, axis=0)
    return final_array


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
