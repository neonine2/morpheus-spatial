import os
from lightning.pytorch import seed_everything

seed_everything(42)

import morpheus as mp


def main(
    data_path,
    additional_cols,
    optimization_param,
    patch_size=16,
    pixel_size=3,
    cell_types=["Tcytotoxic", "Tumor"],
    mask_cell_types=["Tcytotoxic"],
    stratify_by="Contains_Tcytotoxic",
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
        model_path (str): path to the trained model
        model_arch (str): model architecture
        trainer_params (dict): training parameters
        optimization_param (dict): optimization parameters for counterfactual generation
        cf_dir (str): directory to save counterfactuals
    """
    # load data
    dataset = mp.SpatialDataset(input_path=data_path, additional_cols=additional_cols)

    # generate masked patches
    dataset.generate_masked_patch(
        cell_to_mask=mask_cell_types,
        cell_types=cell_types,
        patch_size=patch_size,
        pixel_size=pixel_size,
        save=True,
    )

    print("Loading data...")
    dataset.generate_data_splits(stratify_by=stratify_by)

    # initialize model
    n_channels = dataset.n_channels
    img_size = dataset.img_size
    model_path = dataset.model_path if model_path is None else model_path

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
        (dataset.metadata["Contains_Tumor"] == 1)
        & (dataset.metadata["Contains_Tcytotoxic"] == 0)
        & (dataset.metadata["splits"] == "train")
    ].sample(frac=1, random_state=42)

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
        model_kwargs={"in_channels": n_channels, "img_size": img_size},
    )


if __name__ == "__main__":
    BASE = "/groups/mthomson/zwang2/IMC/output/hochMelanoma_sz48_pxl3_nc41/final"  # change to your own directory
    optimization_param = {
        "use_kdtree": True,
        "theta": 60.0,
        "kappa": -0.7,
        "learning_rate_init": 0.1,
        "beta": 5,
        "max_iterations": 1000,
        "c_init": 25000.0,
        "c_steps": 5,
        "threshold": 0.31,
        "numerical_diff": False,
        "channel_to_perturb": [
            "CCL4_mRNA",
            "CCL18_mRNA",
            "CXCL8_mRNA",
            "CXCL10_mRNA",
            "CXCL12_mRNA",
            "CXCL13_mRNA",
            "CCL2_mRNA",
            "CCL22_mRNA",
            "CXCL9_mRNA",
            "CCL8_mRNA",
            "CCL19_mRNA",
        ],
    }
    main(
        data_path=f"{BASE}/singlecell.csv",
        additional_cols=["Cancer_Stage", "IHC_T_score"],
        cf_dir=f"{BASE}/cf/run_5",
        optimization_param=optimization_param,
    )
