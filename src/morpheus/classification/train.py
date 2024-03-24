import os

from lightning import Trainer
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint, TQDMProgressBar

from ..datasets.spatial_dataset import SpatialDataset
from ..datasets.torch_dataset import make_torch_dataloader
from .classifier import PatchClassifier
from ..configuration import DefaultFolderName


def train(
    model: PatchClassifier,
    dataset: SpatialDataset,
    label_name: str,
    save_model_dir=None,
    dataloader_params=None,
    trainer_params=None,
):
    if trainer_params is None:
        trainer_params = {
            "max_epochs": 100,
            "accelerator": "auto",
            "devices": "auto",
            "logger": False,
        }
    if dataloader_params is None:
        dataloader_params = {"batch_size": 128, "num_workers": 4, "pin_memory": True}
    if save_model_dir is None:
        save_model_dir = os.path.join(dataset.root_dir, DefaultFolderName.model.value)

    # initialize dataloaders
    train_loader, val_loader, test_loader = make_torch_dataloader(
        dataset.split_dir,
        label_name=label_name,
        model_arch=model.arch,
        params=dataloader_params,
    )

    # initialize trainer
    trainer = Trainer(
        devices=1,
        callbacks=[
            ModelCheckpoint(
                monitor="val_bmc",
                mode="max",
                save_top_k=1,
                save_weights_only=False,
                verbose=False,
            ),
            EarlyStopping(
                monitor="val_bmc", min_delta=0, patience=15, verbose=False, mode="max"
            ),
            TQDMProgressBar(refresh_rate=10),
        ],
        default_root_dir=save_model_dir,
        **trainer_params,
    )

    # training
    print(f"Training model with {model.arch} architecture")
    trainer.fit(model=model, train_dataloaders=train_loader, val_dataloaders=val_loader)
    print(f"model saved to {save_model_dir}")
    dataset.model_dir = save_model_dir

    # testing
    trainer.test(ckpt_path="best", dataloaders=test_loader)

    return model
