import os
import sys
from lightning import Trainer
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint, TQDMProgressBar

from src.morpheus.datasets.preprocessing import get_stratified_splits
from src.morpheus.datasets.DataClass import make_torch_dataloader, set_seed
from src.morpheus.models.ModelClass import TissueClassifier

# # Function for setting the seed
set_seed(42)


# def train_model(
#     modelArch, img_dir, patient_info_dir, save_model_dir, split_param, train_params
# ):
#     datasplit_path = get_stratified_splits(img_dir = img_dir, 
#                                            patient_dir=patient_info_dir, 
#                                            param=split_param)
    
#     # train_loader, val_loader, test_loader = make_torch_dataloader(
#     #     datasplit_path, model=modelArch, params=train_params
#     # )

#     # if save_model_dir is None:
#     #     save_model_dir = os.path.join(datasplit_path, f"model/{modelArch}")

#     # # initialize model
#     # model = TissueClassifier(in_channels, img_size, modelArch=modelArch)

#     # trainer = Trainer(
#     #     accelerator="gpu",
#     #     devices=1,
#     #     precision=16,
#     #     max_epochs=100,
#     #     callbacks=[
#     #         ModelCheckpoint(
#     #             monitor="val_bmc",
#     #             mode="max",
#     #             save_top_k=1,
#     #             save_weights_only=True,
#     #             verbose=False,
#     #         ),
#     #         EarlyStopping(
#     #             monitor="val_bmc", min_delta=0, patience=7, verbose=False, mode="max"
#     #         ),
#     #         TQDMProgressBar(refresh_rate=10),
#     #     ],
#     #     default_root_dir=save_model_dir,
#     # )

#     # # training
#     # trainer.fit(model, train_loader, val_loader)
#     # print(f"model saved to {save_model_dir}")

#     # # disable randomness, dropout, etc...
#     # model.eval()

#     # # testing
#     # trainer.test(dataloaders=test_loader)


# if __name__ == "__main__":
#     modelArch = "unet"
#     in_channels = 44
#     img_size = 16
#     train_params = {"batch_size": 64 * 2, "num_workers": 4, "pin_memory": True}
#     split_param = {"eps": 0.01, "train_lb": 0.58, "split_ratio": [0.63, 0.16, 0.21]}

#     img_dir = "/groups/mthomson/zwang2/IMC/output/cedarsLiver_sz48_pxl3_nc44/temp/patched.dat"
#     patient_info_dir = "/groups/mthomson/zwang2/IMC/output/cedarsLiver_sz48_pxl3_nc44/temp/patient_info.csv"

#     train_model(
#         modelArch,
#         img_dir,
#         patient_info_dir,
#         save_model_dir=None,
#         split_param=split_param,
#         train_params=train_params,
#     )
