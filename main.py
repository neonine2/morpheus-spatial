import morpheus as mp
# from morpheus.train import train_classifier

if __name__ == "__main__":
    in_channels = 44
    img_size = 16
    train_params = {"batch_size": 64 * 2, "num_workers": 4, "pin_memory": True}
    split_param = {"eps": 0.01, "train_lb": 0.58, "split_ratio": [0.63, 0.16, 0.21]}

    data_path = "/groups/mthomson/zwang2/IMC/output/cedarsLiver_sz48_pxl3_nc44/temp/crc.h5"
    livertumor = mp.SpatialDataset(data_path)
    livertumor.generate_data_splits(stratify_by='Tcytotoxic', train_size=0.7, test_size=0.15, val_size=0.15)

    # train classifier
    # train_classifier(livertumor, model_arch='unet')

    
