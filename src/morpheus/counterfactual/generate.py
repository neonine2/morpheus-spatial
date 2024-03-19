import os
import numpy as np
import torch
from .cf import Counterfactual

EPSILON = torch.tensor(1e-20, dtype=torch.float32)


def generate_cf(
    original_patch: np.ndarray,
    original_label: np.ndarray,
    model: torch.nn.Module,
    channel_to_perturb: list,
    data_dict: dict,
    X_train_path: str = None,
    optimization_params: dict = None,
    save_dir: str = None,
    patch_id: int = None,
    threshold: float = 0.5,
) -> None:
    """
    Generate counterfactuals for a given image patch.

    Args:
         original_patch (np.ndarray): Original patch to be explained.
         original_label (np.ndarray): Original label of the patch.
         model (torch.nn.Module): Model to be explained.
         channel_to_perturb (list): List of channels to perturb.
         data_dict (dict): Dictionary containing the mean and standard deviation of each channel.
         X_train_path (str, optional): Path to the training data. Defaults to None.
         optimization_params (dict, optional): Dictionary containing the parameters for the optimization. Defaults to {}.
         save_dir (str, optional): Directory where output will be saved. Defaults to None.
         patch_id (int, optional): Patch ID. Defaults to None.
         threshold (float, optional): Threshold for the prediction probability. Defaults to 0.5.

     Returns:
         None
    """
    # Obtain data features
    channel, sigma, mu = (
        np.array(data_dict["channel"]),
        torch.from_numpy(data_dict["stdev"]).float(),
        torch.from_numpy(data_dict["mean"]).float(),
    )
    H, _, C = original_patch.shape
    original_patch = (torch.from_numpy(original_patch).float() - mu) / sigma
    original_label = torch.from_numpy(original_label).long()
    X_mean = torch.mean(original_patch, dim=(0, 1))

    if model.arch == "mlp":
        original_patch = X_mean

    # Adding init layer to model
    unnormed_mean = X_mean * sigma + mu
    if model.arch == "mlp":
        altered_model = lambda x: torch.nn.functional.softmax(model(x), dim=1)
        input_transform = lambda x: x
    else:
        print("Modifying model")
        unnormed_patch = original_patch[None, :] * sigma + mu
        init_fun = lambda y: alter_image(y, unnormed_patch, mu, sigma, unnormed_mean)
        altered_model, input_transform = add_init_layer(init_fun, model)

    # Set range of each channel to perturb
    channel_to_perturb = [name for name in channel if name in channel_to_perturb]
    is_perturbed = np.array(
        [True if name in channel_to_perturb else False for name in channel]
    )
    feature_range = (torch.maximum(-mu / sigma, torch.ones(C) * -4), torch.ones(C) * 4)
    feature_range[0][~is_perturbed] = X_mean[~is_perturbed] - EPSILON
    feature_range[1][~is_perturbed] = X_mean[~is_perturbed] + EPSILON

    # define predict function
    predict_fn = lambda x: altered_model(x)

    # Terminate if model incorrectly classifies patch as the target class
    target_class = optimization_params.pop("target_class")
    orig_proba = predict_fn(X_mean[None, :])
    print(f"Initial probability: {orig_proba}")
    pred = orig_proba[0, 1] > threshold
    if pred == target_class:
        print("Instance already classified as target class, no counterfactual needed")
        return

    # define counterfactual object
    print("defining counterfactual object")
    shape = (1,) + original_patch.shape
    cf = Counterfactual(
        predict_fn,
        input_transform,
        shape,
        feature_range=feature_range,
        **optimization_params,
    )

    print("Building kdtree")
    if not os.path.exists(optimization_params["trustscore"]):
        if X_train_path is None:
            raise ValueError(
                "X_train_path must be provided if trustscore file does not exist."
            )
        X_train = np.load(X_train_path)
        X_train = (X_train - mu) / sigma
        if model.arch == "mlp":
            X_t = torch.from_numpy(np.mean(X_train, axis=(1, 2))).float()
        else:
            X_t = torch.permute(torch.from_numpy(X_train), (0, 3, 1, 2)).float()
        preds = np.argmax(model(X_t).detach().numpy(), axis=1)
        X_train = torch.mean(X_train, dim=(1, 2))
        cf.fit(X_train, preds)
    else:
        cf.fit()

    print("kdtree built!")
    explanation = cf.explain(
        X=X_mean[None, :], Y=original_label[None, :], target_class=[target_class]
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

        print(f"Counterfactual probability: {cf_prob}")
        print(f"Computed probability: {counterfactual_probabilities}")
        X_perturbed = mean_preserve_dimensions(
            cf * sigma + mu, preserveAxis=cf.ndim - 1
        )
        original_patch = X_mean * sigma + mu
        cf_delta = (X_perturbed - original_patch) / original_patch * 100
        print(f"cf delta: {cf_delta}")
        cf_perturbed = dict(zip(channel[is_perturbed], cf_delta[is_perturbed].numpy()))
        print(f"cf perturbed: {cf_perturbed}")

        if save_dir is not None:
            os.makedirs(save_dir, exist_ok=True)
            saved_file = os.path.join(save_dir, f"patch_{patch_id}.npz")
            np.savez(
                saved_file,
                explanation=explanation,
                cf_perturbed=cf_perturbed,
                channel_to_perturb=channel_to_perturb,
            )
    return explanation


def alter_image(y, unnormed_patch, mu, sigma, unnormed_mean):
    unnormed_y = y * sigma + mu
    new_patch = unnormed_patch * ((unnormed_y / unnormed_mean)[:, None, None, :])
    return (new_patch - mu) / sigma


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
