import os
from lightning import Trainer
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint, TQDMProgressBar

from .classifier import PatchClassifier
from ..datasets.torch_dataset import make_torch_dataloader
from ..datasets.spatial_dataset import SpatialDataset


def train(
    model: PatchClassifier,
    dataset: SpatialDataset,
    labelname: str,
    save_model_dir=None,
    dataloader_params={"batch_size": 128, "num_workers": 4, "pin_memory": True},
    trainer_params={
        "max_epochs": 80,
        "accelerator": "auto",
        "logger": False,
    },
):

    if save_model_dir is None:
        save_model_dir = os.path.join(os.path.dirname(dataset.data_path), "models")

    # initialize dataloaders
    train_loader, val_loader, test_loader = make_torch_dataloader(
        dataset.save_dir,
        labelname=labelname,
        model_arch=model.model_arch,
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
                save_weights_only=True,
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
    print(f"Training model with {model.model_arch} architecture")
    trainer.fit(model=model, train_dataloaders=train_loader, val_dataloaders=val_loader)
    print(f"model saved to {save_model_dir}")

    # testing
    trainer.test(ckpt_path="best", dataloaders=test_loader)

    return model
