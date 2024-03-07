import os
from lightning import Trainer
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint, TQDMProgressBar

from .classifier import PatchClassifier
from ..datasets.torch_dataset import make_torch_dataloader
from ..datasets.spatial_dataset import SpatialDataset


def train_classifier(
    dataset: SpatialDataset,
    labelname: str,
    model_arch="unet",
    save_model_dir=None,
    dataloader_params={"batch_size": 64, "num_workers": 4, "pin_memory": True},
    train_params={"max_epochs": 100, "precision": 16, "accelerator": "gpu"},
):

    if save_model_dir is None:
        save_model_dir = os.path.join(os.path.dirname(dataset.data_path), "models")

    # initialize dataloaders
    train_loader, val_loader, test_loader = make_torch_dataloader(
        dataset.save_dir,
        labelname=labelname,
        model_arch=model_arch,
        params=dataloader_params,
    )

    # initialize model
    nchannels = len(dataset.channel_names)
    img_size = dataset.data_dim[1:3]
    model = PatchClassifier(nchannels, img_size, model_arch=model_arch)

    # initialize trainer
    trainer = Trainer(
        devices=1,
        callbacks=[
            ModelCheckpoint(
                monitor="val_bmc",
                mode="max",
                save_top_k=1,
                save_weights_only=True,
                verbose=False,
            ),
            EarlyStopping(
                monitor="val_bmc", min_delta=0, patience=7, verbose=False, mode="max"
            ),
            TQDMProgressBar(refresh_rate=10),
        ],
        default_root_dir=save_model_dir,
        **train_params,
    )

    # training
    print(f"Training model with {model_arch} architecture")
    # trainer.fit(model, train_loader, val_loader)
    # print(f"model saved to {save_model_dir}")

    # # disable randomness, dropout, etc...
    # model.eval()

    # # testing
    # trainer.test(dataloaders=test_loader)

    return model
