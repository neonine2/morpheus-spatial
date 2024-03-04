import os
import sys
import pickle
import numpy as np
import pandas as pd
from tqdm import tqdm
import torch

from ..constants import celltype, splits, colname
from pprint import pprint

def get_stratified_splits(
    img_dir: str,
    patient_dir: str,
    patient_split={},
    overwrite=False,
    save_path=None,
    param={
        "eps": 0.01,
        "train_lb": 0.65,
        "split_ratio": [0.65, 0.15, 0.2],
        "celltype": celltype.cd8.value,
        "ntol": 100,
    },
):
    # get folder path of image data as output path being saved to
    if save_path is None:
        save_path = os.path.dirname(img_dir)

    # generate data split if not already done or overwrite set to True
    if not os.path.isdir(os.path.join(save_path, splits.train.value)) or overwrite:
        print(f"Generating data splits and saving to {save_path} ...")
        stratified_data_split(
            img_dir,
            patient_dir,
            save_path=save_path,
            patient_split=patient_split,
            **param,
        )
    else:
        print(f"Given data directory already created: {save_path}")
        pprint(describe_data_split(save_path, celltype=param['celltype']))
    return save_path


def describe_data_split(save_path, celltype=celltype.cd8.value):
    y_mean = {}
    split_info = np.load(save_path + "/split_info.pkl", allow_pickle=True)
    for split in [splits.train.value, splits.validate.value, splits.test.value]:
        data = pd.read_csv(os.path.join(save_path, f"{split}/label.csv"))
        print(data)
        n_pat = len(split_info[split + "_patient"])
        y = data[celltype].mean()
        y_mean.update({split: [round(y, 3), len(data), n_pat]})
    return y_mean


def stratified_data_split(
    img_dir: str,
    patient_path: str,
    save_path = None,
    patient_split = {},
    celltype = celltype.cd8.value,
    split_ratio=[0.6, 0.2, 0.2],
    eps=0.05,
    train_lb=0.65,
    ntol=100,
):
    predefinedSplit = bool(patient_split)
    if save_path is None:
        save_path = os.path.dirname(img_dir)

    # Ratio of patients in different groups
    train_ratio, valid_ratio, test_ratio = split_ratio

    # load patient and image id
    pat_df = pd.read_csv(patient_path)[[colname.PATIENTID.value, colname.IMAGEID.value]]
    unique_pat_id = np.unique(pat_df[colname.PATIENTID.value])

    # load image data
    try:
        with open(img_dir, "rb") as f:
            intensity, label, channel, _ = pickle.load(f)
    except Exception as e:
        print(f"Error loading image data: {e}")
        return
    
    # split patient into train-test-validation group stratified by T cell level
    npatches = intensity.shape[0]
    isValidSplit = False
    counter = 0
    
    while not isValidSplit and counter < ntol:
        if not predefinedSplit:
            np.random.shuffle(unique_pat_id)
            train_end = int(len(unique_pat_id) * train_ratio)
            valid_end = train_end + int(len(unique_pat_id) * valid_ratio)
            patient_split = {
                splits.train.value: unique_pat_id[:train_end],
                splits.validate.value: unique_pat_id[train_end:valid_end],
                splits.test.value: unique_pat_id[valid_end:],
            }

        # obtain image number corresponding to patient split
        image_split = {
            key: pat_df[pat_df[colname.PATIENTID.value].isin(val)][colname.IMAGEID.value]
            for key, val in patient_split.items()
        }
            
        # shuffle image patch in each split
        patch_split = {}
        label_split = {}
        index_split = {}
        split_balance = {}
        for key, image_ids in image_split.items():
            indices = label[label[colname.IMAGEID.value].isin(image_ids)].index
            shuffled_indices = np.random.permutation(indices)
            patch_split[key] = intensity[shuffled_indices, :]
            label_split[key] = label[celltype].iloc[shuffled_indices]
            index_split[key] = shuffled_indices
            split_balance[key] = label_split[key].mean()

        # compute sample condition values
        tr_prop = patch_split[splits.train.value].shape[0] / npatches
        tr_te_diff = abs(split_balance[splits.train.value] - split_balance[splits.test.value])
        tr_va_diff = abs(split_balance[splits.train.value] - split_balance[splits.validate.value])
        isValidSplit = (tr_te_diff < eps) and (tr_va_diff < eps) and (tr_prop > train_lb)

        # if sample conditions satisfied, save splits
        if isValidSplit or predefinedSplit:
            print("Split constraints satisfied\nPatch proportions and Positive patch proportions:")
            for split, imgs in patch_split.items():
                proportion = imgs.shape[0] / npatches
                positive_proportion = split_balance[split]
                print(f"{split:<10}: {proportion:>5.3f}, {positive_proportion:>5.3f}")

            # save splits
            split_info = {
                "celltype": celltype,
                "channel": channel,
                "patient_df": pat_df,
                "train_set_mean": np.mean(patch_split[splits.train.value], axis=(0, 1, 2)),
                "train_set_stdev": np.std(patch_split[splits.train.value], axis=(0, 1, 2)),
                "patch_shape": intensity.shape[1:],
                "test_patient": patient_split[splits.test.value],
                "validate_patient": patient_split[splits.validate.value],
                "train_patient": patient_split[splits.train.value],
                "test_index": index_split[splits.test.value],
                "validate_index": index_split[splits.validate.value],
                "train_index": index_split[splits.train.value],
            }
            save_splits(save_path, patch_split, label_split, split_info)
            return
        else:
            counter += 1
            print(f"Attempt {counter}: Could not satisfy data split constraints, trying again.")
    print("Could not satisfy data split constraints, try again or adjust constraints")


def save_splits(save_path, data_dict, label_dict, split_info):
    # save split info
    with open(os.path.join(save_path, "split_info.pkl"), "wb") as f:
        pickle.dump(split_info, f, protocol=4)

    # save splits
    for split in tqdm(data_dict.keys(), desc="Saving splits"):
        img_array = data_dict[split]

        # make dir
        _path = os.path.join(save_path, split)
        if not os.path.isdir(_path):
            os.makedirs(_path)
            os.makedirs(os.path.join(_path, "0"))
            os.makedirs(os.path.join(_path, "1"))

        # save labels
        label_dict[split].to_csv(os.path.join(_path, "label.csv"), index=False)

        # save images
        np.save(os.path.join(_path, "img.npy"), img_array)
        nimage = img_array.shape[0]
        patch_label = label_dict[split].values
        patch_index = split_info[f'{split}_index']
        for i in tqdm(range(nimage), desc=f"Saving images for {split} split", leave=False):
            label = patch_label[i]
            index = patch_index[i]
            dense_tensor = torch.tensor(img_array[i, ...])
            sparse_tensor = dense_tensor.to_sparse()
            # Save the sparse tensor
            torch.save(sparse_tensor, os.path.join(_path, f"{label}/patch_{index}.pt"))
