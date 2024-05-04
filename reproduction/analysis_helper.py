import os
import json
import numpy as np
import pandas as pd
import torch
import umap

from morpheus import SpatialDataset, load_model


def retrieve_perturbation(
    dataset: SpatialDataset,
    additional_col: list = [],
) -> pd.DataFrame:
    cf_path = dataset.cf_dir
    # get all npz files in the directory
    npz_files = [f for f in os.listdir(cf_path) if f.endswith(".npz")]

    # reorder the files by the number in the file name
    get_id = lambda x: int(x.split("_")[1].split(".")[0])
    npz_files = sorted(npz_files, key=get_id)

    # combine all cf from npz files
    cf_perturbed = []
    for npz_file in npz_files:
        cf = np.load(os.path.join(cf_path, npz_file), allow_pickle=True)
        perturbation = cf["delta_in_percentage"].item().values()
        cf_proba = cf["proba"][1]
        cf_perturbed.append([get_id(npz_file)] + [cf_proba] + list(perturbation))
    channels_perturbed = [name for name in cf["delta_in_percentage"].item().keys()]

    # create a dataframe
    all_cf = pd.DataFrame(
        cf_perturbed,
        columns=["patch_id", "proba"] + channels_perturbed,
    )

    # zero out minor changes in channels_perturbed column
    all_cf[channels_perturbed] = all_cf[channels_perturbed].map(
        lambda x: 0 if abs(x) < 1 else x
    )

    # combine dataset.metadata with data_tf using the patch_id
    all_cf = all_cf.merge(
        dataset.metadata[["patch_id", "PatientID", "ImageNumber"]],
        on="patch_id",
        how="left",
    )

    # if additional_col is not empty, read the single cell file from dataset.input_path and merge that column using ImageNumber into all_cf
    if additional_col:
        all_cf = get_additional_col(all_cf, dataset, additional_col)

    return all_cf, channels_perturbed


def get_additional_col(df, dataset, colname):
    df = df.merge(
        pd.read_csv(
            dataset.input_path,
            usecols=["ImageNumber"] + colname,
            low_memory=False,
        ).drop_duplicates(),
        on="ImageNumber",
        how="left",
    )
    return df


def filter_images(
    metadata: pd.DataFrame,
    remove_healthy=True,
    remove_small_images=True,
    remove_few_tumor_cells=True,
):
    # for each remove option, update the metadata
    if remove_healthy and "type" in metadata.columns:
        metadata = metadata[metadata["type"] != "Nor"]
    if remove_small_images:
        filter = metadata.groupby("ImageNumber").count()["Contains_Tumor"] >= 16
        metadata = metadata[metadata["ImageNumber"].isin(filter[filter].index)]
    if remove_few_tumor_cells:
        filter = metadata.groupby("ImageNumber").sum()["Contains_Tumor"] > 4
        metadata = metadata[metadata["ImageNumber"].isin(filter[filter].index)]
    return metadata


def apply_perturbation(img_patch, perturbation, channel_names):
    opt_perb = np.zeros(len(channel_names))
    is_perturbed = [name in perturbation.keys() for name in channel_names]
    opt_perb[is_perturbed] = perturbation
    X_perb = img_patch * (1 + opt_perb / 100)
    return X_perb


def load_data_split(
    dataset,
    data_split,
    remove_healthy,
    remove_small_images,
    remove_few_tumor_cells,
    additional_col,
    parallel=False,
):
    # get test data
    _metadata = dataset.metadata[(dataset.metadata["splits"] == data_split)]
    if additional_col:  # add columns like tissue type or FLD
        _metadata = get_additional_col(_metadata, dataset, additional_col)
    original_patch_count = _metadata["ImageNumber"].nunique()

    _metadata = filter_images(
        _metadata,
        remove_healthy=remove_healthy,
        remove_small_images=remove_small_images,
        remove_few_tumor_cells=remove_few_tumor_cells,
    )
    final_image_count = _metadata["ImageNumber"].nunique()
    print("Number of tissues filtered out:", original_patch_count - final_image_count)
    print("Number of tissues remaining:", final_image_count)

    X = dataset.load_from_metadata(_metadata, parallel=parallel)
    y = _metadata["Contains_Tcytotoxic"].values.flatten()
    return X, y, _metadata


def load_classifier(dataset, mu, stdev):
    classifier = load_model(
        dataset.model_path,
        **{"in_channels": dataset.n_channels, "img_size": dataset.img_size},
    )
    wrapped_classifier = (
        lambda x: classifier(
            torch.permute(torch.from_numpy((x - mu) / stdev).float(), (0, 3, 1, 2))
        )
        .detach()
        .numpy()[:, 1]
    )
    return wrapped_classifier


def get_data_and_model(
    dataset: SpatialDataset,
    data_split: str,
    remove_healthy: bool = True,
    remove_small_images: bool = True,
    remove_few_tumor_cells: bool = True,
    additional_col: list = [],
    data_only: bool = False,
    pallalel: bool = False,
):
    # load image data and label
    X, y, test_metadata = load_data_split(
        dataset,
        data_split,
        remove_healthy,
        remove_small_images,
        remove_few_tumor_cells,
        additional_col,
        parallel=pallalel,
    )

    if data_only:
        return X, y, test_metadata

    # Load normalization parameters
    with open(
        os.path.join(dataset.split_dir, "normalization_params.json")
    ) as json_file:
        normalization_params = json.load(json_file)
        mu = normalization_params["mean"]
        stdev = normalization_params["stdev"]

    # load classifier
    model = load_classifier(dataset, mu, stdev)

    return X, y, test_metadata, model


def get_umap_embeddings(
    dataset: SpatialDataset,
    cf_df: pd.DataFrame,
    channel_to_perturb: list,
    data_split: str = "train",
):

    # load data and model
    X, y, test_metadata = get_data_and_model(
        dataset,
        data_split=data_split,
        remove_healthy=False,
        remove_small_images=False,
        remove_few_tumor_cells=False,
        data_only=True,
    )

    # get mean intensity
    mean_intensity = np.mean(X, axis=(1, 2))
    df = pd.DataFrame(mean_intensity, columns=dataset.channel_names)
    df_normalized = np.arcsinh(df.div(df.sum(axis=1), axis=0))

    # remove rows that contain NaN or infinite values
    valid_rows = np.isfinite(df_normalized).all(axis=1).tolist()
    print(f"number of rows removed: {len(df_normalized) - np.array(valid_rows).sum()}")
    df_normalized = df_normalized[valid_rows]
    y = y[valid_rows]
    test_metadata = test_metadata[valid_rows]
    df_normalized.index = test_metadata["patch_id"]

    # get counterfactuals
    print("Loading counterfactuals")
    # orig_cf = np.zeros([len(cf_df), len(dataset.channel_names)])
    # for id, patch_id in enumerate(cf_df["patch_id"].to_numpy()):
    #     img = np.load(f"{dataset.split_dir}/{data_split}/0/patch_{patch_id}.npy")
    #     orig_cf[id, :] = np.mean(img, axis=(0, 1))
    # orig_cf = pd.DataFrame(orig_cf, columns=dataset.channel_names)
    # orig_cf.index = cf_df["patch_id"]

    perturb_df = cf_df[channel_to_perturb] / 100 + 1
    perturb_df.index = cf_df["patch_id"]

    # keep all channel until after row normalization
    # df_orig_normalized = np.arcsinh(
    # orig_cf.div(orig_cf.sum(axis=1), axis=0)[channel_to_perturb]
    # )

    # get indices from df_normalized that are in perturb_df and have it in the same order
    perturbed = df_normalized.loc[perturb_df.index].copy()
    assert perturbed.index.equals(perturb_df.index)  # check if the indices are the same
    assert (
        not perturbed.isnull().values.any()
    )  # throw error if there are any NaN values
    # apply perturbation
    perturbed[channel_to_perturb] = perturb_df * perturbed[channel_to_perturb]
    # normalize
    df_perturbed_normalized = np.arcsinh(perturbed.div(perturbed.sum(axis=1), axis=0))
    # remove rows that contain NaN or infinite values
    valid_rows = np.isfinite(df_perturbed_normalized).all(axis=1)
    print(f"number of invalid rows removed: {np.sum(~valid_rows)}")
    df_perturbed_normalized = df_perturbed_normalized[valid_rows]

    np.random.seed(42)
    # Create a UMAP object with a fixed random state
    umap_model = umap.UMAP(random_state=42, n_jobs=-1, verbose=False)
    # fit UMAP model to the entire training data
    print("Fitting UMAP model")
    embedding = umap_model.fit_transform(df_normalized[channel_to_perturb])
    # transform the original and perturbed data
    print("Computing UMAP embeddings")
    umap_perturbed = umap_model.transform(df_perturbed_normalized[channel_to_perturb])
    umap_orig = umap_model.transform(
        df_normalized.loc[perturb_df.index, channel_to_perturb]
    )

    # combine the embeddings with the labels into one dataframe
    embedding_df = pd.DataFrame(embedding, columns=["umap1", "umap2"])
    embedding_df["patch_id"] = df_normalized.index
    embedding_df = embedding_df.merge(test_metadata, on="patch_id", how="left")

    # combine umap_orig and umap_perturbed with the patch_id and patient_id
    umap_orig = pd.DataFrame(umap_orig, columns=["orig_umap1", "orig_umap2"])
    umap_orig["patch_id"] = perturbed.index

    # add umap_perturbed to umap_orig
    umap_perturbed = pd.DataFrame(
        umap_perturbed, columns=["perturbed_umap1", "perturbed_umap2"]
    )

    # combine umap_orig and umap_perturbed
    umap_cf = pd.concat(
        [umap_orig.reset_index(drop=True), umap_perturbed.reset_index(drop=True)],
        axis=1,
    )

    return embedding_df, umap_cf


def assess_perturbation(
    dataset: SpatialDataset,
    perturbation: pd.DataFrame,
    data_split: str = "test",
    classify_threshold: float = 0.5,
    remove_healthy: bool = True,
    remove_small_images: bool = True,
    remove_few_tumor_cells: bool = True,
    additional_col: list = [],
):
    # load data and model (including normalization parameters)
    X_test, y_test, test_metadata, model = get_data_and_model(
        dataset,
        data_split,
        remove_healthy,
        remove_small_images,
        remove_few_tumor_cells,
        additional_col,
    )

    # predict the original image patch
    pred_orig = model(X_test) > classify_threshold
    print("original (predicted) = %.3f" % (pred_orig).mean())
    print("original (true) = %.3f" % y_test.mean())

    # perturb image patch by iterating over each row of perturbation
    pred_perturbed_list = []
    for i in range(len(perturbation)):
        perturbed_patch = apply_perturbation(
            X_test, perturbation.iloc[i], dataset.channel_names
        )
        pred_perturbed = model(perturbed_patch) > classify_threshold
        pred_perturbed_list.append(pred_perturbed)
        print(f"perturbed {i} = {(pred_perturbed).mean():.3f}")

    # map each patch to patient
    pre_post_df = pd.DataFrame(
        {
            "patch_id": test_metadata["patch_id"],
            "true_orig": y_test,
            "pred_orig": pred_orig,
            "strategy_1": pred_perturbed_list[0],
            "strategy_2": pred_perturbed_list[1],
        }
    )
    pre_post_df = pre_post_df.merge(test_metadata, on="patch_id", how="left")

    return pre_post_df