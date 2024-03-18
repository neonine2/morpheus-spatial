from lightning.pytorch import seed_everything

import src.morpheus as mp

if __name__ == "__main__":
    # set random seed
    seed_everything(42, workers=True)

    # STEP 1: Create a spatial dataset object
    data_path = "data/crc.h5"  # path to the spatial dataset
    label_name = "Tcytotoxic"

    # initialize spatial dataset object
    livertumor = mp.SpatialDataset(data_path)

    # STEP 2: Generate data splits to prepare for model training
    # generate data splits
    livertumor.generate_data_splits(
        stratify_by=label_name, train_size=0.7, test_size=0.15, val_size=0.15
    )

    # STEP 3: Train a PyTorch model
    # initialize model
    model_arch = "unet"
    n_channels = len(livertumor.channel_names)
    img_size = livertumor.data_dim[1:3]
    model = mp.PatchClassifier(n_channels, img_size, model_arch=model_arch)

    # train model
    model = mp.train(model=model, dataset=livertumor, label_name=label_name)

    # STEP 4: Evaluate the trained model
    # TODO: Add evaluation code
