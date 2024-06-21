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
    predict_label: str,
    save_model_dir=None,
    dataloader_params=None,
    trainer_params=None,
    only_train=False,
):
    if trainer_params is None:
        trainer_params = {
            "max_epochs": 30,
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
        label_name=predict_label,
        model_arch=model.arch,
        params=dataloader_params,
    )

    # Setup the checkpoint callback
    checkpoint_callback = ModelCheckpoint(
        monitor="val_bmc",
        mode="max",
        save_top_k=1,
        save_weights_only=False,
        verbose=False,
        dirpath=save_model_dir,
    )

    # initialize trainer
    trainer = Trainer(
        devices=1,
        callbacks=[
            checkpoint_callback,
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
    dataset.model_path = checkpoint_callback.best_model_path
    print(f"model saved to {dataset.model_path}")

    # testing
    if not only_train:
        trainer.test(ckpt_path=dataset.model_path, dataloaders=test_loader)

    return model


def test_model(
    dataset: SpatialDataset,
    predict_label: str = 'Contains_Tcytotoxic',
    model_arch: str = "unet",
    model_path: str = None,
    dataloader_params=None,
    **model_kwargs,
):
    if dataloader_params is None:
        dataloader_params = {"batch_size": 128, "num_workers": 4, "pin_memory": True}
    if model_path is None:
        model_path = dataset.model_path

    # load model
    model = PatchClassifier.load_from_checkpoint(
        checkpoint_path=model_path, **model_kwargs
    )

    # initialize dataloaders
    _, _, test_loader = make_torch_dataloader(
        dataset.split_dir,
        label_name=predict_label,
        model_arch=model_arch,
        params=dataloader_params,
    )

    # testing
    trainer = Trainer(devices=1, logger=False)
    print(f"Testing model at {model_path}")
    trainer.test(model=model, dataloaders=test_loader)

    return
