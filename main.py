import os
from lightning.pytorch import seed_everything

import src.morpheus as mp

if __name__ == "__main__":
    # set random seed
    seed_everything(42, workers=True)

    # STEP 1: Create a spatial dataset object
    root_dir = "/groups/mthomson/zwang2/IMC/output/cedarsLiver_sz48_pxl3_nc44/temp"  # change to your own directory
    data_path = os.path.join(root_dir, "crc.h5")

    # initialize spatial dataset object
    livertumor = mp.SpatialDataset(data_path=data_path)

    # STEP 2: Generate data splits to prepare for model training
    # generate data splits
    label_name = "Tcytotoxic"
    livertumor.generate_data_splits(stratify_by=label_name)

    # STEP 3: Train a PyTorch model
    # # initialize model
    # model_arch = "unet"
    # n_channels = livertumor.n_channels
    # img_size = livertumor.img_size
    # model = mp.PatchClassifier(n_channels, img_size, arch=model_arch)

    # # train model
    # trainer_params = {
    #     "max_epochs": 2,
    #     "accelerator": "auto",
    #     "logger": False,
    # }
    # model = mp.train(
    #     model=model,
    #     dataset=livertumor,
    #     label_name=label_name,
    #     trainer_params=trainer_params,
    # )

    # images to generate counterfactuals
    select_metadata = livertumor.metadata[
        (livertumor.metadata["Tumor"] == 1)
        & (livertumor.metadata["Tcytotoxic"] == 0)
        & (livertumor.metadata["splits"] == "train")
    ]
    # channels allowed to be perturbed
    channel_to_perturb = [
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
    ]

    # threshold for classification
    threshold = 0.5

    # optimization parameters
    optimization_param = {
        "use_kdtree": True,
        "theta": 40.0,
        "kappa": 0,  # set to: (threshold - 0.5) * 2
        "learning_rate_init": 0.1,
        "beta": 40.0,
        "max_iterations": 10,
        "c_init": 1000.0,
        "c_steps": 5,
    }

    # load model if needed
    model_path = os.path.join(
        livertumor.model_dir, "checkpoints/epoch=49-step=17400.ckpt"
    )
    model = livertumor.load_model(model_path, arch="unet")

    # Generate counterfactuals using trained model
    cf = mp.get_counterfactual(
        images=select_metadata.iloc[:10],
        dataset=livertumor,
        target_class=1,
        model=model,
        channel_to_perturb=channel_to_perturb,
        optimization_params=optimization_param,
        threshold=threshold,
        num_workers=1,
    )
