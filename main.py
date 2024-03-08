import morpheus as mp
from morpheus.train import train_classifier
from lightning.pytorch import seed_everything

if __name__ == "__main__":
    # set random seed
    seed_everything(42, workers=True)

    data_path = (
        "/groups/mthomson/zwang2/IMC/output/cedarsLiver_sz48_pxl3_nc44/temp/crc.h5"
    )
    labelname = "Tcytotoxic"

    # initialize spatial dataset object
    livertumor = mp.SpatialDataset(data_path)

    # generate data splits
    livertumor.generate_data_splits(
        stratify_by=labelname, train_size=0.7, test_size=0.15, val_size=0.15
    )

    # train classifier
    train_classifier(livertumor, model_arch="unet", labelname=labelname)
