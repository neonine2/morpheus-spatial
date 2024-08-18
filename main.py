import os
from lightning.pytorch import seed_everything

seed_everything(42)

import morpheus as mp
import json


def main(
    data_path,
    optimization_param,
    additional_cols=[],
    patch_size=16,
    pixel_size=3,
    cd8_name="Tcytotoxic",
    tumor_name="Tumor",
    stratify_by=None,
    mask_cell_types=None,
    patient_split=None,
    model_path=None,
    model_arch="unet",
    trainer_params={"max_epochs": 30, "accelerator": "auto", "logger": False},
    cf_dir=None,
):
    """
    Main function to train model and generate counterfactuals for dataset

    Args:
        data_path (str): path to the dataset
        additional_cols (list): additional columns to include in the dataset
        patch_size (int): patch size in pixels
        pixel_size (int): pixel size in microns
        cell_types (list): list of cell types
        mask_cell_types (list): list of cell types to mask
        stratify_by (str): column to stratify by when splitting the dataset
        patient_split (dict): specify patient split
        model_path (str): path to the trained model
        model_arch (str): model architecture
        trainer_params (dict): training parameters
        optimization_param (dict): optimization parameters for counterfactual generation
        cf_dir (str): directory to save counterfactuals
    """
    # load data
    dataset = mp.SpatialDataset(input_path=data_path, additional_cols=additional_cols)

    # generate masked patches
    mask_cell_types = [cd8_name] if mask_cell_types is None else mask_cell_types
    dataset.generate_masked_patch(
        cell_to_mask=mask_cell_types,
        cell_types=[cd8_name, tumor_name],
        patch_size=patch_size,
        pixel_size=pixel_size,
        save=True,
    )

    print("Loading data...")
    stratify_by = f"Contains_{cd8_name}" if stratify_by is None else stratify_by
    dataset.generate_data_splits(stratify_by=stratify_by, specify_split=patient_split)

    # initialize model
    n_channels = dataset.n_channels
    img_size = dataset.img_size
    model_path = dataset.model_path if model_path is None else model_path
    print("Model path:", model_path)

    if model_path is None:
        model = mp.PatchClassifier(n_channels, img_size, model_arch)
        # train model
        print("Training model...")
        model = mp.train(
            model=model,
            dataset=dataset,
            predict_label=stratify_by,
            trainer_params=trainer_params,
        )
        model_path = dataset.model_path  # saved model path
    else:
        print(f"Loading model from {model_path}...")

    # images to generate counterfactuals
    dataset.get_split_info()
    select_metadata = dataset.metadata[
        (dataset.metadata[f"Contains_{tumor_name}"] == 1)
        & (dataset.metadata[f"Contains_{cd8_name}"] == 0)
        & (dataset.metadata["splits"] == "train")
    ]

    # example of selected instances to generate counterfactuals
    print(select_metadata.head())

    # Generate counterfactuals using trained model
    print("Generating counterfactuals...")
    save_dir = f"{dataset.root_dir}/cf/" if cf_dir is None else cf_dir
    mp.get_counterfactual(
        images=select_metadata,
        dataset=dataset,
        target_class=1,
        model_path=model_path,
        optimization_params=optimization_param,
        save_dir=save_dir,
        device="cpu",
        num_workers=os.cpu_count() - 1,
        verbosity=0,
        tumor_name=tumor_name,
        cd8_name=cd8_name,
    )


if __name__ == "__main__":
    # Example of running the main function on a data set
    optimization_param = {
        "use_kdtree": True,
        "theta": 50.0,
        "kappa": -0.34,
        "learning_rate_init": 0.1,
        "beta": 80.0,
        "max_iterations": 1000,
        "c_init": 1000.0,
        "c_steps": 5,
        "channel_to_perturb": [
            "Glnsynthetase",
            "CCR4",
            "PDL1",
            "LAG3",
            "CD105endoglin",
            "TIM3",
            "CXCR4",
            "PD1",
            "CYR61",
            "CD44",
            "IL10",
            "CXCL12",
            "CXCR3",
            "Galectin9",
            "YAP",
        ],
    }
    BASE = "crc"  # change to your own directory
    # For paper reproduction purpose: load patient split and trained model
    with open(os.path.join(BASE, "patient_split.json"), "r") as file:
        patient_split = json.load(file)
    main(
        data_path=os.path.join(BASE, "singlecell.csv"),
        additional_cols=["type", "FLD"],
        cf_dir=os.path.join(BASE, "cf/example"),
        optimization_param=optimization_param,
        patient_split=patient_split,
    )
