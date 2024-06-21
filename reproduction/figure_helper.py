import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib import colors, cm
import matplotlib.lines as mlines
import scipy.cluster.hierarchy as sch
from sklearn.linear_model import LinearRegression
from random import shuffle
from scipy.stats import t
from analysis_helper import load_data_split, get_data_and_model
import h5py
from matplotlib.ticker import ScalarFormatter


def plot_prediction_scatterplot(pred_df: pd.DataFrame, save_fig: bool = False):
    _, ax = plt.subplots(figsize=(6, 6))
    color = ["deeppink", "forestgreen", "slateblue"]
    x = np.linspace(0, 1, 100)
    for i, (name, pred) in enumerate(pred_df.items()):
        x = pred["true"].to_numpy()
        y = pred["pred_binary"].to_numpy()

        # create and fit the linear regression model
        linmodel = LinearRegression()
        X = x.reshape(-1, 1)
        linmodel.fit(X, y)
        r_sq = linmodel.score(X, y)

        ax.scatter(x, y, c=color[i], label=name)
        ax.plot(x, linmodel.predict(x.reshape(-1, 1)), c=color[i])
        plt.text(
            0.7,
            0.19 - i * 0.08,
            "R² = {:.2f}".format(r_sq),
            fontsize=18,
            c=color[i],
        )

    ax.legend(frameon=False, fontsize=14)
    ax.plot([0, 1], [0, 1], c="black")
    # set aspect ratio to 1
    plt.xlabel("True proportion of patches with T cells", fontsize=16)
    plt.ylabel("Predict proportion of patches with T cells", fontsize=16)

    # increase tick label font size
    ax.tick_params(axis="both", which="major", labelsize=13)

    if save_fig:
        plt.savefig(
            "prediction_scatterplot.svg", format="svg", dpi=300, bbox_inches="tight"
        )
    plt.show()


def plot_rmse(all_rmse, save_fig: bool = False):
    # Create a sample DataFrame
    rmse = {
        "U-Net": [all_rmse["Melanoma"], all_rmse["CRC"], all_rmse["Breast tumor"]],
        "MLP": [0.12, 0.106, 0.113],
        "Linear": [0.15, 0.17, 0.16],
    }
    df = pd.DataFrame(rmse, index=["Melanoma", "CRC", "Breast tumor"])
    # plot grouped bar chart
    _ = df.plot(kind="bar", figsize=(6, 6))

    # add title and labels
    plt.xlabel("Tumor Type", fontsize=16)
    plt.xticks(rotation=0)
    plt.ylabel("Root Mean Squared Error (RMSE)", fontsize=16)

    # move legend outside the plot to the right
    plt.legend(loc="center left", bbox_to_anchor=(1, 0.5), fontsize=14)

    # increase tick label font size
    plt.tick_params(axis="both", which="major", labelsize=16)

    if save_fig:
        plt.savefig("rmse_barplot.svg", format="svg", dpi=300, bbox_inches="tight")

    # show the plot
    plt.show()


def get_upper_thresh(mla_data):
    X, _, _metadata = load_data_split(
        mla_data,
        data_split="train",
        remove_healthy=False,
        remove_small_images=False,
        remove_few_tumor_cells=False,
        parallel=False,
    )
    q75, q25 = np.percentile(
        np.mean(X, axis=(1, 2)), [75, 25], axis=0
    )  # determine thres based on IQR of original data
    return np.max((q75 - q25) / q25) * 1.5 * 100


def plot_patient_perturbation(
    dataset,
    rel_perturbation: pd.DataFrame,
    channel_to_perturb: list,
    numClust: int = 2,
    extra_cols: list = [],
    save_fig: bool = False,
    set_thresh: bool = True,
    keep_top_2: bool = False,
):
    """
    Plot the perturbation of each patient as well as the median perturbation of each cluster of patients
    """
    # Remove normal tissue samples (for crc dataset only)
    if "type" in rel_perturbation.columns:
        rel_perturbation = rel_perturbation[rel_perturbation["type"] != "Nor"]

    # Filter out rows with extreme perturbations
    if set_thresh:
        thresh = get_upper_thresh(dataset)  # set thresh based on IQR of original data
    else:  # precomputed for example dataset
        thresh = 79 * 100
    upper_bound = np.maximum(
        np.median(rel_perturbation[channel_to_perturb].quantile(0.99)), thresh
    )  # use maximum to be conservative with filter
    lower_bound = np.minimum(
        np.median(rel_perturbation[channel_to_perturb].quantile(0.00)), -101
    )
    to_keep = (rel_perturbation[channel_to_perturb] <= upper_bound).all(axis=1) & (
        rel_perturbation[channel_to_perturb] >= lower_bound
    ).all(axis=1)
    rel_perturbation = rel_perturbation[to_keep]

    # Plot the perturbation of each patient
    patient_cf = (
        rel_perturbation.groupby("PatientID")
        .agg(
            {
                **{k: "median" for k in channel_to_perturb},
                **{
                    k: "first"
                    for k in rel_perturbation.columns
                    if k not in channel_to_perturb
                },
            }
        )
        .drop("PatientID", axis=1)
    )

    grouped_cf = patient_cf[channel_to_perturb]
    # remove rows with all zeros
    grouped_cf = grouped_cf.loc[(grouped_cf != 0).any(axis=1)]
    grouped_cf = grouped_cf.dropna(how="all", axis=1)  # remove columns with all NaN

    # Create a custom colormap that maps zero to white
    cmap = sns.diverging_palette(240, 10, as_cmap=True)  # "vlag" colormap
    cmap.set_bad("white")  # Set color for zeros to white

    vmin = grouped_cf.min().min()
    vmax = grouped_cf.max().max()
    axe = sns.clustermap(
        grouped_cf,
        row_cluster=True,
        col_cluster=False,
        cmap=cmap,
        method="average" if vmax > 500 else "complete",
        metric="correlation",
        xticklabels=[x[:-5] if x.endswith("_mRNA") else x for x in channel_to_perturb],
        # yticklabels=False,
        norm=colors.TwoSlopeNorm(vmin=vmin, vcenter=0, vmax=vmax / 5 + 1),
        figsize=(8, 8),
    )

    if save_fig:
        axe.savefig("clustermap.svg", format="svg", dpi=300, bbox_inches="tight")

    # if FLD or Cancer_Stage is in patient_cf, merge them into grouped_cf using PatientID
    if extra_cols:
        grouped_cf = grouped_cf.join(
            patient_cf[extra_cols],
            on="PatientID",
            how="left",
        )
    clusters = sch.fcluster(axe.dendrogram_row.linkage, numClust, "maxclust")
    grouped_cf["cluster"] = clusters

    # Plot the median perturbation of each cluster of patients
    if keep_top_2:
        top_two_clusters = grouped_cf["cluster"].value_counts().nlargest(2).index
        grouped_cf = grouped_cf[grouped_cf["cluster"].isin(top_two_clusters)]
        grouped_cf["cluster"] = grouped_cf["cluster"] - 1
        numClust = grouped_cf["cluster"].nunique()

    _, ax = plt.subplots(numClust, 1, figsize=(8, numClust * 1.5))
    strategy_cluster = []
    for ii in range(numClust):
        med_strat = np.median(
            grouped_cf.loc[grouped_cf["cluster"] == ii + 1, channel_to_perturb], axis=0
        )
        med_strat[abs(med_strat) < 5] = 0
        strategy_cluster.append(med_strat)
        ax[ii].bar(
            range(len(channel_to_perturb)),
            strategy_cluster[ii],
            color=[
                "tab:blue" if val < 0 else "tab:red" for val in strategy_cluster[ii]
            ],
        )
        ax[ii].set_yscale("symlog")
        # get rid of the x-axis bar
        ax[ii].xaxis.set_visible(False)

        # draw line as x = 0
        ax[ii].axhline(0, color="black")
        ax[ii].set_ylabel("Median change (%)")

    if save_fig:
        plt.savefig("cluster_median.svg", format="svg", dpi=300, bbox_inches="tight")

    plt.show()

    strategy_cluster = pd.DataFrame(strategy_cluster, columns=channel_to_perturb)

    if extra_cols:
        print(grouped_cf.groupby("cluster")[extra_cols].value_counts())

    return strategy_cluster, grouped_cf


def aggregate_performance_per_patient(tcell_level, patient_phenotype=None):
    patient_phenotype = "splits" if patient_phenotype is None else patient_phenotype
    # if 'type' is in tcell_level, add {"type": "first"} to the groupby
    agg = {
        "true_orig": "mean",
        "strategy_1": "mean",
        "strategy_2": "mean",
        "PatientID": "first",
        patient_phenotype: "first",
    }
    if "type" in tcell_level.columns:
        agg.update({"type": "first"})

    tcell_level_image = tcell_level.groupby("ImageNumber").agg(agg)

    tcell_level_patient = tcell_level_image.groupby("PatientID").agg(
        {
            "true_orig": "median",
            "strategy_1": "median",
            "strategy_2": "median",
            patient_phenotype: "first",
        }
    )
    # get the 1st and 3rd quartile of the T cell infiltration level
    if patient_phenotype == "Cancer_Stage":  # only for melanoma dataset
        for col in ["true_orig", "strategy_1", "strategy_2"]:
            tcell_level_patient[col + "_q1"] = tcell_level_image.groupby("PatientID")[
                col
            ].quantile(0.25)
            tcell_level_patient[col + "_q3"] = tcell_level_image.groupby("PatientID")[
                col
            ].quantile(0.75)
    elif patient_phenotype == "splits":
        for col in ["true_orig", "strategy_1", "strategy_2"]:
            tcell_level_patient[col + "_q1"] = tcell_level_image.groupby("PatientID")[
                col
            ].quantile(0.25)
            tcell_level_patient[col + "_q3"] = tcell_level_image.groupby("PatientID")[
                col
            ].quantile(0.75)

    return tcell_level_patient, tcell_level_image


def plot_perturbation_performance(
    tcell_level: pd.DataFrame,
    patient_phenotype: str = None,
    strategy_mapping: dict = None,
    strat2_color="#2ba02c",
    strat1_color="#faaf40",
    save_fig: bool = False,
):
    # Aggregate performance per patient
    tcell_level_patient, tcell_level_image = aggregate_performance_per_patient(
        tcell_level, patient_phenotype
    )

    # map each patient to a strategy based on the patient_phenotype and strategy_mapping
    if strategy_mapping is not None:
        tcell_level_patient["pred_perturbed"] = tcell_level_patient.apply(
            lambda x: (
                x["strategy_1"]
                if x[patient_phenotype] == strategy_mapping["strategy_1"]
                else x["strategy_2"]
            ),
            axis=1,
        )
        tcell_level_image["pred_perturbed"] = tcell_level_image.apply(
            lambda x: (
                x["strategy_1"]
                if x[patient_phenotype] == strategy_mapping["strategy_1"]
                else x["strategy_2"]
            ),
            axis=1,
        )

    # group the tcell_level_image by type and patientID and take the median of true_orig and pred_perturbed
    tcell_level_image = tcell_level_image.reset_index()

    if patient_phenotype == "FLD":  # only for crc dataset
        make_line_plots(
            tcell_level_image,
            patient_phenotype,
            strategy_mapping,
            strat1_color,
            strat2_color,
            save_fig,
        )

        tcell_level_image = (
            tcell_level_image.groupby(["PatientID", "type"])
            .agg(
                {
                    "true_orig": "median",
                    "pred_perturbed": "median",
                    "ImageNumber": "first",
                }
            )
            .reset_index()
        )

        tcell_level_patient = plot_two_vertical_bar(
            tcell_level_patient,
            tcell_level_image,
            patient_phenotype,
            strat1_color,
            strat2_color,
            save_fig,
        )

    elif patient_phenotype == "Cancer_Stage":  # melanoma dataset
        plot_horizontal_bar(
            tcell_level_patient,
            patient_phenotype,
            strategy_mapping,
            strat1_color,
            strat2_color,
            save_fig,
        )

    else:
        plot_multi_horizontal_bar(
            tcell_level_patient,
            strat1_color,
            strat2_color,
            save_fig,
        )

    return 


def plot_multi_horizontal_bar(
    tcell_level_patient,
    strat1_color="#faaf40",
    strat2_color="#2ba02c",
    save_fig: bool = False,
):
    # reorder the columns
    tcell_level_patient = tcell_level_patient[
        [
            "true_orig",
            "strategy_1",
            "strategy_2",
            "true_orig_q1",
            "true_orig_q3",
            "strategy_1_q1",
            "strategy_1_q3",
            "strategy_2_q1",
            "strategy_2_q3",
        ]
    ]

    # subtract the true_orig from the quantiles
    tcell_level_patient["true_orig_q1"] = (
        tcell_level_patient["true_orig"] - tcell_level_patient["true_orig_q1"]
    )
    tcell_level_patient["true_orig_q3"] = (
        tcell_level_patient["true_orig_q3"] - tcell_level_patient["true_orig"]
    )

    # subtract the strategy_1 from the quantiles
    tcell_level_patient["strategy_1_q1"] = (
        tcell_level_patient["strategy_1"] - tcell_level_patient["strategy_1_q1"]
    )
    tcell_level_patient["strategy_1_q3"] = (
        tcell_level_patient["strategy_1_q3"] - tcell_level_patient["strategy_1"]
    )

    # subtract the strategy_2 from the quantiles
    tcell_level_patient["strategy_2_q1"] = (
        tcell_level_patient["strategy_2"] - tcell_level_patient["strategy_2_q1"]
    )
    tcell_level_patient["strategy_2_q3"] = (
        tcell_level_patient["strategy_2_q3"] - tcell_level_patient["strategy_2"]
    )

    # Sort the patients by the true_orig
    tcell_level_patient = tcell_level_patient.sort_values(by="true_orig")
    _, ax = plt.subplots(figsize=(6.2, 5))

    colors = ["gray", strat1_color, strat2_color]
    labels = ["Original", "Strategy 1", "Strategy 2"]

    # Plot the T cell infiltration level of each patient
    bars = tcell_level_patient.plot(
        y=["true_orig", "strategy_1", "strategy_2"],
        kind="barh",
        ax=ax,
        color={
            "true_orig": "gray",
            "strategy_1": strat1_color,
            "strategy_2": strat2_color,
        },
        width=0.8,
    )

    # Adding error bars
    for i, (_, row) in enumerate(tcell_level_patient.iterrows()):
        if (
            row["true_orig_q1"] != row["true_orig_q3"]
            and row["strategy_1_q1"] != row["strategy_1_q3"]
        ):
            ax.errorbar(
                x=[row["true_orig"], row["strategy_1"], row["strategy_2"]],
                y=[
                    i - 0.2,
                    i,
                    i + 0.2,
                ],  # Adjust these positions based on your specific bar layout
                xerr=[
                    [
                        tcell_level_patient.loc[_, "true_orig_q1"],
                        tcell_level_patient.loc[_, "strategy_1_q1"],
                        tcell_level_patient.loc[_, "strategy_2_q1"],
                    ],  # Lower errors
                    [
                        tcell_level_patient.loc[_, "true_orig_q3"],
                        tcell_level_patient.loc[_, "strategy_1_q3"],
                        tcell_level_patient.loc[_, "strategy_2_q3"],
                    ],
                ],  # Upper errors
                fmt="none",  # This removes any connecting lines
                color="black",  # Error bar color
                capsize=3,  #
            )

    # Moving the x-axis to the top of the plot
    ax.xaxis.tick_top()  # Moves the ticks to the top
    ax.xaxis.set_label_position("top")  # Moves the x-axis label to the top

    # Move the x-axis line to the top
    ax.spines["bottom"].set_visible(False)  # Hides the bottom spine
    ax.spines["top"].set_visible(True)  # Makes sure the top spine is visible
    ax.spines["top"].set_position(
        ("outward", 0)
    )  # Moves the top spine to the edge of the plot

    plt.xlabel("T cell infiltration level")
    plt.ylabel("Test patients")

    ax.set_yticks(range(len(tcell_level_patient)))
    ax.set_yticklabels([])
    legend_handles = [plt.Rectangle((0, 0), 1, 1, color=color) for color in colors]
    ax.legend(legend_handles, labels, loc="lower right")

    plt.tight_layout()
    if save_fig:
        plt.savefig(
            "efficacy_hbar_plot.svg", format="svg", dpi=300, bbox_inches="tight"
        )
    plt.show()


def plot_horizontal_bar(
    tcell_level_patient,
    patient_phenotype,
    strategy_mapping=None,
    strat1_color="#faaf40",
    strat2_color="#2ba02c",
    save_fig: bool = False,
):

    # map the quantiles as well
    tcell_level_patient["pred_perturbed_q1"] = tcell_level_patient.apply(
        lambda x: (
            x["strategy_1"] - x["strategy_1_q1"]
            if x[patient_phenotype] == strategy_mapping["strategy_1"]
            else x["strategy_2_q1"] - x["strategy_2_q1"]
        ),
        axis=1,
    )
    tcell_level_patient["pred_perturbed_q3"] = tcell_level_patient.apply(
        lambda x: (
            x["strategy_1_q3"] - x["strategy_1"]
            if x[patient_phenotype] == strategy_mapping["strategy_1"]
            else x["strategy_2_q3"] - x["strategy_2"]
        ),
        axis=1,
    )
    tcell_level_patient["true_orig_q1"] = (
        tcell_level_patient["true_orig"] - tcell_level_patient["true_orig_q1"]
    )
    tcell_level_patient["true_orig_q3"] = (
        tcell_level_patient["true_orig_q3"] - tcell_level_patient["true_orig"]
    )

    tcell_level_patient = tcell_level_patient.drop(
        columns=[
            "strategy_1",
            "strategy_2",
            "strategy_1_q1",
            "strategy_1_q3",
            "strategy_2_q1",
            "strategy_2_q3",
        ]
    )

    # reorder the columns
    tcell_level_patient = tcell_level_patient[
        [
            "pred_perturbed",
            "true_orig",
            patient_phenotype,
            "true_orig_q1",
            "true_orig_q3",
            "pred_perturbed_q1",
            "pred_perturbed_q3",
        ]
    ]

    # Sort the patients by the true_orig
    tcell_level_patient = tcell_level_patient.sort_values(by="true_orig")
    _, ax = plt.subplots(figsize=(6.2, 5))

    colors = ["gray", strat1_color, strat2_color]
    labels = ["Original", "Strategy 1", "Strategy 2"]

    # Plot the T cell infiltration level of each patient
    bars = tcell_level_patient.plot(
        y=["true_orig", "pred_perturbed"],
        kind="barh",
        ax=ax,
        color={"true_orig": "gray", "pred_perturbed": [strat1_color, strat2_color]},
        width=0.8,
    )

    # Adding error bars
    for i, (_, row) in enumerate(tcell_level_patient.iterrows()):
        # only add error bars if both q1 and q3 are not zero
        if (
            row["true_orig_q1"] != 0
            and row["true_orig_q3"] != 0
            and row["pred_perturbed_q1"] != 0
            and row["pred_perturbed_q3"] != 0
        ):
            ax.errorbar(
                x=[row["true_orig"], row["pred_perturbed"]],
                y=[
                    i - 0.2,
                    i + 0.2,
                ],  # Adjust these positions based on your specific bar layout
                xerr=[
                    [
                        tcell_level_patient.loc[_, "true_orig_q1"],
                        tcell_level_patient.loc[_, "pred_perturbed_q1"],
                    ],  # Lower errors
                    [
                        tcell_level_patient.loc[_, "true_orig_q3"],
                        tcell_level_patient.loc[_, "pred_perturbed_q3"],
                    ],
                ],  # Upper errors
                fmt="none",  # This removes any connecting lines
                color="black",  # Error bar color
                capsize=3,  # Error bar cap size at the top
            )

    # Moving the x-axis to the top of the plot
    ax.xaxis.tick_top()  # Moves the ticks to the top
    ax.xaxis.set_label_position("top")  # Moves the x-axis label to the top

    # Move the x-axis line to the top
    ax.spines["bottom"].set_visible(False)  # Hides the bottom spine
    ax.spines["top"].set_visible(True)  # Makes sure the top spine is visible
    ax.spines["top"].set_position(
        ("outward", 0)
    )  # Moves the top spine to the edge of the plot

    plt.xlabel("T cell infiltration level")
    plt.ylabel("Test patients")

    ax.set_yticks(range(len(tcell_level_patient)))
    ax.set_yticklabels([])

    legend_handles = [plt.Rectangle((0, 0), 1, 1, color=color) for color in colors]
    ax.legend(legend_handles, labels, loc="lower right")

    plt.tight_layout()
    if save_fig:
        plt.savefig(
            "efficacy_hbar_plot.svg", format="svg", dpi=300, bbox_inches="tight"
        )
    plt.show()


def plot_two_vertical_bar(
    tcell_level_patient,
    tcell_level_image,
    patient_phenotype,
    strat1_color,
    strat2_color,
    save_fig: bool = False,
):
    tcell_level_patient = tcell_level_patient.drop(
        columns=[
            "strategy_1",
            "strategy_2",
        ]
    )
    # make two vertical bar plots, one for patientID with 18, 13, 22, 28 and one for patientID with 17, 7, 14
    _, ax = plt.subplots(2, 1, figsize=(4.5, 6))

    colors = ["gray", strat1_color, strat2_color]
    labels = ["Original", "Strategy 1", "Strategy 2"]

    # if patient_phenotype is FLD, color the bars based on the strategy, strat1_color for FLD = 1 and strat2_color for FLD = 0
    tcell_level_patient = tcell_level_patient.reset_index()
    tcell_level_patient["color"] = tcell_level_patient[patient_phenotype].apply(
        lambda x: strat1_color if x == 1 else strat2_color
    )
    tcell_level_image = tcell_level_image.merge(
        tcell_level_patient[["PatientID", "color"]], on="PatientID"
    )

    for i, patientID in enumerate([[18, 13, 22, 28], [17, 7, 14]]):
        if i == 0:
            tcell_level_patient_subset = tcell_level_patient[
                tcell_level_patient["PatientID"].isin(patientID)
            ]
            tcell_level_patient_subset = tcell_level_patient_subset.sort_values(
                by="true_orig"
            )

            # Plot the T cell infiltration level of each patient
            bars = tcell_level_patient_subset.plot(
                y=["true_orig", "pred_perturbed"],
                kind="bar",
                ax=ax[i],
                color={
                    "true_orig": "gray",
                    "pred_perturbed": tcell_level_patient_subset["color"],
                },
                width=0.8,
            )

            # x tick label upright
            ax[i].set_xticklabels(
                tcell_level_patient_subset["PatientID"],
                rotation=0,
                horizontalalignment="center",
            )

            ax[i].set_ylabel("T cell infiltration level")
            ax[i].set_xlabel("Test patients")

            # legend only for i = 1
            if i == 1:
                legend_handles = [
                    plt.Rectangle((0, 0), 1, 1, color=color) for color in colors
                ]
                ax[i].legend(legend_handles, labels, loc="upper left")
            else:
                ax[i].get_legend().remove()
        elif i == 1:
            # plot type = 'metaT' and type = 'PriT' separately for the second plot

            tcell_level_image_subset = tcell_level_image[
                tcell_level_image["PatientID"].isin(patientID)
            ]
            # reorder the rows so the type = 'metaT' is plotted first
            tcell_level_image_subset = tcell_level_image_subset.sort_values(by=["type"])
            bars = tcell_level_image_subset.plot(
                y=["true_orig", "pred_perturbed"],
                kind="bar",
                ax=ax[i],
                color={
                    "true_orig": "gray",
                    "pred_perturbed": tcell_level_image_subset["color"],
                },
                width=0.8,
            )

            # x tick label upright
            ax[i].set_xticklabels(
                tcell_level_image_subset["PatientID"],
                rotation=0,
                horizontalalignment="center",
            )

            ax[i].set_ylabel("T cell infiltration level")
            ax[i].set_xlabel("Test patients")

            # legend only for j = 1
            if i == 1:
                legend_handles = [
                    plt.Rectangle((0, 0), 1, 1, color=color) for color in colors
                ]
                ax[i].legend(legend_handles, labels, loc="upper left")
            else:
                ax[i].get_legend().remove()

    plt.tight_layout()
    if save_fig:
        plt.savefig("efficacy_bar_plot.svg", format="svg", dpi=300, bbox_inches="tight")
    plt.show()

    return tcell_level_image_subset


def make_line_plots(
    tcell_level_image,
    patient_phenotype,
    strategy_mapping,
    strat1_color,
    strat2_color,
    save_fig: bool = False,
):
    # make a separate line plot for each patient, each line in a plot is a separate image with two points for original and perturbed T cell infiltration level
    patient_id = tcell_level_image["PatientID"].unique()
    tcell_level_image["color"] = tcell_level_image[patient_phenotype].apply(
        lambda x: strat1_color if x == strategy_mapping["strategy_1"] else strat2_color
    )

    # make a grid of line plots, each plot is for a patient, each line in a plot is a separate image with two points for original and perturbed T cell infiltration level
    num_plots = len(patient_id)
    num_cols = 6
    num_rows = int(np.ceil(num_plots / num_cols))

    # add more space between the plots
    fig, axes = plt.subplots(num_rows, num_cols, figsize=(10, 2.5 * num_rows))
    fig.subplots_adjust(hspace=0.5, wspace=0.7)

    for i, pid in enumerate(patient_id):
        row_idx = i // num_cols
        col_idx = i % num_cols
        tcell_level_image_subset = tcell_level_image[
            tcell_level_image["PatientID"] == pid
        ]
        for _, row in tcell_level_image_subset.iterrows():
            ax = axes[row_idx, col_idx]
            if row["type"] == "metaT":
                ls = "solid"
            elif row["type"] == "PriT":
                ls = "dashed"
            # make a line plot for each image with a colored point for original and perturbed T cell infiltration level,
            ax.plot(
                [0, 1],
                [row["true_orig"], row["pred_perturbed"]],
                color=row["color"],
                linestyle=ls,
                marker="o",
                label=row["type"],
            )
            # make the true_orig point gray
            ax.plot(
                0,
                row["true_orig"],
                color="gray",
                marker="o",
            )

            ax.set_title(f"Patient {pid}")
            ax.set_xticks([0, 1])
    plt.tight_layout()
    if save_fig:
        plt.savefig(
            "efficacy_line_plot.svg", format="svg", dpi=300, bbox_inches="tight"
        )
    plt.show()


def plot_umap_embedding_crc(embedding_df, umap_cf, save_fig: bool = False):
    plt.figure(figsize=(8, 5))
    noTcell = (embedding_df["Contains_Tcytotoxic"] == 0) & (
        embedding_df["Contains_Tumor"] == 1
    )
    plt.scatter(
        x=embedding_df.loc[noTcell, "umap1"],
        y=embedding_df.loc[noTcell, "umap2"],
        s=0.2,
        c="tab:red",
        alpha=0.5,
        label="no T cells (with tumor)",
    )
    cmap = {"Nor": "tab:purple", "PriT": "tab:cyan", "metaT": "goldenrod"}
    label_for_legend = {
        "Nor": "T cells (healthy tissue)",
        "PriT": "T cells (primary)",
        "metaT": "T cells (metastatic)",
    }
    for key, value in cmap.items():
        cond = (embedding_df["Contains_Tcytotoxic"] == 1) & (
            embedding_df["type"] == key
        )
        plt.scatter(
            x=embedding_df.loc[cond, "umap1"],
            y=embedding_df.loc[cond, "umap2"],
            s=0.2,
            c=value,
            label=label_for_legend[key],
            alpha=0.8,
        )

    # Create legend with larger markers using proxy artists
    legend_elements = [
        mlines.Line2D(
            [0],
            [0],
            color="tab:red",
            marker="o",
            linestyle="None",
            markersize=5,
            label="Tumor patch without T cell",
        ),
        mlines.Line2D(
            [0],
            [0],
            color="tab:purple",
            marker="o",
            linestyle="None",
            markersize=5,
            label="Healthy tissue",
        ),
        mlines.Line2D(
            [0],
            [0],
            color="goldenrod",
            marker="o",
            linestyle="None",
            markersize=5,
            label="Metastatic tumor",
        ),
        mlines.Line2D(
            [0],
            [0],
            color="tab:cyan",
            marker="o",
            linestyle="None",
            markersize=5,
            label="Primary tumor",
        ),
    ]

    plt.legend(handles=legend_elements, fontsize=10)

    plt.axis("off")  # Turn off the axis
    ax = plt.gca()
    ax.invert_xaxis()
    ax.invert_yaxis()
    if save_fig:
        plt.savefig("umap_embedding_crc.png", dpi=300, bbox_inches="tight")
    plt.show()

    plt.figure(figsize=(8, 5), facecolor="white")
    plt.scatter(
        x=umap_cf["orig_umap1"],
        y=umap_cf["orig_umap2"],
        s=0.2,
        c="#EE4B2B",
        label="No T cells",
        alpha=0.8,
    )

    plt.scatter(
        x=umap_cf["perturbed_umap1"],
        y=umap_cf["perturbed_umap2"],
        s=0.2,
        c="tab:blue",
        label="Perturbed",
        alpha=0.8,
    )

    for i in range(0, len(umap_cf)):
        plt.arrow(
            umap_cf["orig_umap1"][i],
            umap_cf["orig_umap2"][i],
            umap_cf["perturbed_umap1"][i] - umap_cf["orig_umap1"][i],
            umap_cf["perturbed_umap2"][i] - umap_cf["orig_umap2"][i],
            color="tab:blue",
            alpha=0.01,
            head_width=0.1,
            length_includes_head=True,
        )

    # Create legend with larger markers using proxy artists
    legend_elements = [
        mlines.Line2D(
            [0],
            [0],
            color="#EE4B2B",
            marker="o",
            linestyle="None",
            markersize=5,
            label="Tumor patch without T cell",
        ),
        mlines.Line2D(
            [0],
            [0],
            color="tab:blue",
            marker="o",
            linestyle="None",
            markersize=5,
            label="Counterfactual",
        ),
    ]
    plt.legend(handles=legend_elements, fontsize=10)
    ax = plt.gca()
    ax.invert_xaxis()
    ax.invert_yaxis()
    plt.axis("off")  # Turn off the axis
    # add white background
    if save_fig:
        plt.savefig("umap_embedding_cf_crc.png", dpi=300, bbox_inches="tight")
    plt.show()


def plot_umap_embedding(
    embedding_df, umap_cf, pie_chart: bool = False, save_fig: bool = False
):
    cond1 = (embedding_df["Contains_Tcytotoxic"] == 0) & (
        embedding_df["Contains_Tumor"] == 1
    )

    cond2 = (embedding_df["Contains_Tcytotoxic"] == 1) & (
        embedding_df["Contains_Tumor"] == 1
    )

    plt.figure(figsize=(8, 5))

    plt.scatter(
        x=embedding_df.loc[cond1, "umap1"],
        y=embedding_df.loc[cond1, "umap2"],
        s=0.2,
        c="#FF0000",
        alpha=0.5,
    )
    plt.scatter(
        x=embedding_df.loc[cond2, "umap1"],
        y=embedding_df.loc[cond2, "umap2"],
        s=0.2,
        c="#04b497",
        alpha=0.8,
    )

    # Create legend with larger markers using proxy artists
    legend_elements = [
        mlines.Line2D(
            [0],
            [0],
            color="#FF0000",
            marker="o",
            linestyle="None",
            markersize=5,
            label="without T cells",
        ),
        mlines.Line2D(
            [0],
            [0],
            color="#04b497",
            marker="o",
            linestyle="None",
            markersize=5,
            label="with T cells",
        ),
    ]

    plt.legend(handles=legend_elements)
    plt.axis("off")  # Turn off the axis
    if save_fig:
        plt.savefig("umap_embedding.png", dpi=300, bbox_inches="tight")
    plt.show()

    # group the perturbed points into clusters
    from sklearn.cluster import KMeans

    x = np.array(umap_cf[["perturbed_umap1", "perturbed_umap2"]])

    # Create a KMeans object with 2 clusters
    ncluster = 8
    kmeans = KMeans(n_clusters=ncluster, random_state=42)

    # Fit the KMeans object to the data
    kmeans.fit(x)

    # Get the labels of the clusters
    cluster_labels = kmeans.labels_
    label_counts = np.bincount(cluster_labels)
    # print(label_counts)

    plt.figure(figsize=(8, 5))
    plt.scatter(
        x=umap_cf["orig_umap1"],
        y=umap_cf["orig_umap2"],
        s=0.2,
        c="#FF0000",
        label="No T cells",
        alpha=0.5,
    )

    # Draw arrows from each point in dataset 1 to its corresponding point in dataset 2
    color_map = [
        "tab:blue",
        "tab:orange",
        "darkgray",
        "darkgray",
        "tab:purple",
        "darkgray",
        "darkgray",
        "tab:green",
    ]

    for i in range(0, len(cluster_labels), 4):
        alpha = 0.05
        if cluster_labels[i] in np.argsort(label_counts)[-2:]:
            alpha = 0.025
        elif color_map[cluster_labels[i]] == "tab:orange":
            alpha = 0.08
        elif color_map[cluster_labels[i]] == "darkgray":
            alpha = 0
        elif color_map[cluster_labels[i]] in ["tab:blue", "tab:green"]:
            alpha = 0.12
        plt.arrow(
            umap_cf["orig_umap1"][i],
            umap_cf["orig_umap2"][i],
            umap_cf["perturbed_umap1"][i] - umap_cf["orig_umap1"][i],
            umap_cf["perturbed_umap2"][i] - umap_cf["orig_umap2"][i],
            color=color_map[cluster_labels[i]],
            alpha=alpha,
            head_width=0.01,
            length_includes_head=True,
        )

    plt.scatter(
        x=umap_cf["perturbed_umap1"],
        y=umap_cf["perturbed_umap2"],
        s=0.3,
        c=[color_map[ii] for ii in cluster_labels],
        label="Counterfactual",
        alpha=[1 if color_map[ii] != "darkgray" else 0 for ii in cluster_labels],
    )
    plt.axis("off")  # Turn off the axis
    if save_fig:
        plt.savefig("umap_embedding_cf.png", dpi=300, bbox_inches="tight")
    plt.show()

    if pie_chart:
        df = embedding_df.loc[cond2]
        centroids = kmeans.cluster_centers_
        # get the index of the 4 largest clusters
        largest_clusters = np.argsort(label_counts)[-4:]

        def generate_distinct_colors(num_colors):
            cmap = plt.get_cmap("nipy_spectral", num_colors * 2)
            return [cmap(i * 2) for i in range(num_colors)]

        # Create a consistent color map for PatientIDs
        unique_patients = df["PatientID"].unique()
        colors = generate_distinct_colors(len(unique_patients))
        shuffle(colors)
        color_map = {patient: color for patient, color in zip(unique_patients, colors)}

        # Set up the figure and axes for the subplots
        fig, axes = plt.subplots(1, 4, figsize=(10, 10 / 4))  # 1 row, 4 columns

        for i, cluster_ in enumerate(largest_clusters):
            near_centroid = (
                np.sqrt(
                    np.sum((df[["umap1", "umap2"]] - centroids[cluster_]) ** 2, axis=1)
                )
                < 0.4
            )
            nearPatient = df.loc[near_centroid, "PatientID"].value_counts()

            # Map the colors for the current pie chart
            pie_colors = [color_map[patient] for patient in nearPatient.index]

            # Plot on the i-th subplot axis
            axes[i].pie(
                nearPatient,
                labels=nearPatient.index,
                startangle=90,
                colors=pie_colors,
            )
            axes[i].set_title(f"cluster centroid {np.round(centroids[cluster_])}")

        # Adjust layout to prevent overlap
        plt.tight_layout()

        # Save figure if needed
        if save_fig:
            plt.savefig("pie_charts.svg", dpi=300, bbox_inches="tight")

        plt.show()


def plot_data(ax, plot_df, names, p_cutoff, strat2_color, strat1_color):
    significant_mask = plot_df["-log10(p_value_adj)"] > -np.log10(p_cutoff)
    positive_fc_mask = plot_df["log2(fold_change)"] > 0
    negative_fc_mask = plot_df["log2(fold_change)"] < 0

    # Plot significant positive fold changes
    ax.scatter(
        plot_df.loc[significant_mask & positive_fc_mask, "log2(fold_change)"],
        plot_df.loc[significant_mask & positive_fc_mask, "-log10(p_value_adj)"],
        color=strat1_color,
        alpha=0.7,
    )

    # Plot significant negative fold changes
    ax.scatter(
        plot_df.loc[significant_mask & negative_fc_mask, "log2(fold_change)"],
        plot_df.loc[significant_mask & negative_fc_mask, "-log10(p_value_adj)"],
        color=strat2_color,
        alpha=0.7,
    )

    # Plot non-significant points
    ax.scatter(
        plot_df.loc[~significant_mask, "log2(fold_change)"],
        plot_df.loc[~significant_mask, "-log10(p_value_adj)"],
        color="gray",
        alpha=0.5,
    )

    ax.axvline(x=0, color="black", linestyle="--")  # Vertical line at x=0
    if not significant_mask.all():
        ax.axhline(
            y=-np.log10(p_cutoff),
            color="black",
            linestyle="--",
            label="Significance Threshold",
        )  # Significance threshold

    # Adding text labels to points
    for i in plot_df.index:
        ax.text(
            plot_df.loc[i, "log2(fold_change)"] * 1.01,
            plot_df.loc[i, "-log10(p_value_adj)"] * 1.01,
            (
                names[i][:-5] if "_mRNA" in names[i] else names[i]
            ),  # remove _mRNA if present
            fontsize=10,
            ha="left",
        )
    ax.set_yscale("log")
    ax.tick_params(axis="both", which="major", labelsize=12)


def make_volcano_plot(
    gene_df,
    celltype_df,
    p_cutoff=0.05,
    strat2_color="#2ba02c",
    strat1_color="#faaf40",
    save_fig: bool = False,
):
    """
    Create a volcano plot to visualize differential expression results.

    Parameters:
    - plot_df (pd.DataFrame): DataFrame containing log2 fold changes and adjusted p-values for the differential analysis.
    - compare (str, default="gene"): Determines the label for the plot, could be "gene" or "celltype".
    - p_cutoff (float, default=0.05): Significance threshold for highlighting significant results on the plot.
    - save_volcanoplot (bool, default=False): If True, the plot will be saved to the specified path.

    Returns:
    - None: Displays and optionally saves a volcano plot.
    """
    # Setup the figure and subplots
    fig, axes = plt.subplots(
        nrows=1, ncols=2, figsize=(6.3, 3.65)
    )  # Adjust the figure size as needed

    # Plot the first dataframe
    plot_data(axes[0], gene_df, gene_df["gene"], p_cutoff, strat2_color, strat1_color)

    # remove unclassified celltypes
    celltype_df = celltype_df[celltype_df["celltype"] != "Unclassified"].reset_index()

    # Plot the second dataframe
    plot_data(
        axes[1],
        celltype_df,
        celltype_df["celltype"],
        p_cutoff,
        strat2_color,
        strat1_color,
    )

    # Set a common xlabel and ylabel
    fig.text(
        0.5,
        0,
        "Log2 (patient cluster 1 / patient cluster 2)",
        ha="center",
        va="center",
        fontsize=14,
    )  # Central x-axis label
    fig.text(
        0,
        0.5,
        "-Log10 (Adjusted P-value)",
        ha="center",
        va="center",
        rotation="vertical",
        fontsize=14,
    )  # Central y-axis label
    # set ylim
    plt.tick_params(axis="both", which="major", labelsize=12)
    plt.tight_layout()  # Adjust subplots to fit into figure area.
    if save_fig:
        plt.savefig("volcano_plot.svg", format="svg", dpi=300, bbox_inches="tight")
    plt.show()


def plot_tissue_level_perturbation(cf_df, channel_to_perturb, save_fig: bool = False):

    nrow_per_image = cf_df.groupby("ImageNumber").size()

    image_to_perturb = nrow_per_image[nrow_per_image > 1].index

    # get the perturbations for the images
    image_perturbation = cf_df[cf_df["ImageNumber"].isin(image_to_perturb)]

    # create a new column named label that is a combination of ImageNumber and type
    image_perturbation = image_perturbation.copy()
    image_perturbation.loc[:, "label"] = (
        image_perturbation["PatientID"].astype(str)
        + "_"
        + image_perturbation["type"].astype(str)
    )

    # apply median to all channel_to_perturb_crc and then "first" to the column named type
    df = image_perturbation.groupby("label").agg(
        {
            **{
                k: "median" for k in channel_to_perturb
            },  # apply median to all channel_to_perturb_crc
            "type": "first",  # apply "first" to the column named type
        }
    )

    cmap = sns.diverging_palette(240, 10, as_cmap=True)  # "vlag" colormap
    cmap.set_bad("white")  # Set color for zeros to white

    # Separate the type column for use in coloring
    types = df["type"]

    # Create a color map for the 'type' column
    type_colors = {"Nor": "tab:purple", "metaT": "goldenrod", "PriT": "tab:cyan"}
    row_colors = types.map(type_colors)

    vmin = df[channel_to_perturb].min().min()
    vmax = df[channel_to_perturb].max().max()

    # First, create a clustermap to get the order of the rows
    initial_cluster = sns.clustermap(
        df[channel_to_perturb],
        row_cluster=True,
        col_cluster=True,
        cmap=cmap,
        method="ward",
        xticklabels=channel_to_perturb,
        norm=colors.TwoSlopeNorm(vmin=vmin, vcenter=0, vmax=vmax / 3 + 1),
        figsize=(10, 10),
    )
    # dont show plot
    plt.close()

    # Reorder row_colors according to the dendrogram's leaves
    ordered_row_colors = row_colors.iloc[initial_cluster.dendrogram_row.reordered_ind]

    # Create the heatmap
    vmin = df[channel_to_perturb].min().min()
    vmax = df[channel_to_perturb].max().max()
    axe = sns.clustermap(
        df[channel_to_perturb],
        row_cluster=True,
        col_cluster=False,
        yticklabels=False,
        cmap=cmap,
        method="ward",
        xticklabels=channel_to_perturb,
        norm=colors.TwoSlopeNorm(vmin=vmin, vcenter=0, vmax=vmax / 3 + 1),
        row_colors=ordered_row_colors,
        figsize=(10, 10),
    )
    if save_fig:
        plt.savefig(
            "tissue_level_heatmap.svg", format="svg", dpi=300, bbox_inches="tight"
        )
    plt.show()


def make_patch_heatmap_mla(dfSUBSET, labelSUBSET, save_fig: bool = False):
    # Create a new clustermap with only the subset of columns clustered
    norm = colors.TwoSlopeNorm(vcenter=0.0)

    ax = sns.clustermap(
        dfSUBSET,
        z_score=0,
        row_cluster=True,
        col_cluster=False,
        cmap="PuOr_r",
        method="ward",
        xticklabels=[name_[:-5] for name_ in dfSUBSET.columns],
        yticklabels=[],
        norm=norm,
        figsize=(8, 10),
    )
    ax.ax_heatmap.set_xticklabels(
        ax.ax_heatmap.get_xticklabels(), rotation=80, fontsize=16
    )
    ax.cax.tick_params(labelsize=16)
    row_order = ax.dendrogram_row.reordered_ind

    # fig = ax.figure
    # fig.savefig(f'patch_clustermap.png', dpi=300)

    # Access the Axes objects associated with the clustermap
    ax_heatmap = ax.ax_heatmap
    ax_col_dendrogram = ax.ax_col_dendrogram
    ax_row_dendrogram = ax.ax_row_dendrogram
    if save_fig:
        plt.savefig("patch_clustermap.png", dpi=300)
    plt.show()
    # Remove the heatmap from the plot
    # ax_heatmap.remove()

    # Save the dendrogram and row/column labels as a vector PDF file
    # fig = ax.figure
    # fig.savefig(f"patch_plot.svg", dpi=300, rasterized=False)

    data = labelSUBSET.iloc[row_order, :]  # reorder the data
    fig, axes = plt.subplots(
        nrows=1,
        ncols=3,
        sharex=True,
        sharey=True,
        gridspec_kw={"wspace": 0, "hspace": 0},
    )
    cmaplist = ["YlGnBu", "RdPu", "PuOr"]
    for i, col in enumerate(["Contains_Tcytotoxic", "Contains_Tumor", "IHC_T_score"]):
        val = data[col].values
        if col == "IHC_T_score":
            val = np.array([[1] if item == "I" else [0] for item in val])
        else:
            val = val.astype(float)
        axes[i].imshow(
            val[:, np.newaxis], cmap="YlGnBu_r", aspect=0.002, interpolation="spline36"
        )
        axes[i].set_xticklabels([])
        axes[i].set_yticklabels([])
        # remove the x and y axis
        axes[i].axis("off")
    # set the margins to zero
    plt.subplots_adjust(
        left=0.05, right=0.95, bottom=0.05, top=0.95, wspace=0, hspace=0
    )
    if save_fig:
        plt.savefig("patch_heatmap.png", dpi=300)


def make_multi_strat_plot(
    allStrategy, tcell_level_crc, chemlabel, save_fig: bool = False
):

    # get column names that contain the word 'strategy'
    strategy_cols = ["true_orig"] + tcell_level_crc.columns[
        tcell_level_crc.columns.str.contains("strategy")
    ].tolist()
    img_mean = tcell_level_crc.groupby(["ImageNumber"]).agg(
        {col: "mean" for col in strategy_cols}
    )

    infiltrate_mean = img_mean.mean()

    # Calculate Standard Error of the Mean (SEM)
    sem = np.std(img_mean) / np.sqrt(img_mean.count())

    # Confidence Interval Bounds
    infiltrate_lower = infiltrate_mean - t.ppf(0.975, df=img_mean.count() - 1) * sem
    infiltrate_upper = infiltrate_mean + t.ppf(0.975, df=img_mean.count() - 1) * sem

    # Error for plotting: [Lower bounds, Upper bounds]
    error = np.array(
        [infiltrate_mean - infiltrate_lower, infiltrate_upper - infiltrate_mean]
    )

    plt.figure(figsize=(9, 7))
    plt.subplot(211, aspect=7)
    plt.bar(
        np.arange(len(allStrategy)),
        infiltrate_mean[1:],
        yerr=error[:, 1:],
        align="center",
        capsize=3,
        color="#17a2b8",
        width=0.6,
        label="Perturbed",
    )
    plt.axhline(
        infiltrate_mean["true_orig"],
        color="tab:gray",
        linewidth=1.5,
        ls="--",
        label="Original",
    )

    # Add a transparent rectangle around the horizontal line to represent the error
    rect = plt.Rectangle(
        (-0.5, infiltrate_lower["true_orig"]),
        4,
        infiltrate_upper["true_orig"] - infiltrate_lower["true_orig"],
        facecolor="tab:gray",
        alpha=0.5,
    )
    plt.ylabel("T cell infiltration level")
    plt.ylim([0.1, 0.82])

    ax = plt.gca()
    ax.add_patch(rect)
    # ax.set_yticks(np.arange(0.1, 0.6, 0.1))
    # current_values = ax.get_yticks()
    # ax.set_yticklabels(["{:,.0%}".format(x) for x in current_values])
    plt.legend(frameon=False, fontsize="12")

    plt.subplot(212, aspect=4)
    # select a divergent colormap
    norm = colors.TwoSlopeNorm(
        vmin=np.min(allStrategy), vcenter=0.0, vmax=np.max(allStrategy) + 1
    )
    # remove _mRNA from the column names if present
    allStrategy.columns = [col.split("_")[0] for col in allStrategy.columns]
    plt.imshow(
        np.transpose(allStrategy).loc[chemlabel], norm=norm, cmap=cm.RdBu_r, aspect=0.5
    )
    ax = plt.gca()
    # Major ticks
    ax.set_xticks(np.arange(0, 5, 1))
    ax.set_yticks(np.arange(0, 10, 1))

    # Minor ticks
    ax.set_xticks(np.arange(-0.5, 5, 1), minor=True)
    ax.set_yticks(np.arange(-0.5, 10, 1), minor=True)

    # Gridlines based on minor ticks
    ax.grid(which="minor", axis="y", color="w", linestyle="-", linewidth=2)
    ax.grid(which="minor", axis="x", color="w", linestyle="-", linewidth=8)

    # remove _mRNA from the column names if present

    plt.yticks(np.arange(len(chemlabel)), np.array(chemlabel), fontsize="8")
    ax.spines.top.set_visible(False)
    ax.spines.left.set_visible(False)
    ax.spines.bottom.set_visible(False)
    ax.spines["right"].set(color="white")
    plt.tick_params(
        axis="both",  # changes apply to the x-axis
        which="both",  # both major and minor ticks are affected
        bottom=False,  # ticks along the bottom edge are off
        top=False,
        left=False,  # ticks along the top edge are off
        labelbottom=False,
    )  # labels along the bottom edge are off
    cbar = plt.colorbar(ticks=[200, 100, 0, -50, -100])
    cbar.ax.set_yticklabels(
        ["$200$", "$100$", "$0$", "$-50$", "$-100$"], fontsize=7
    )  # vertically oriented colorbar
    cbar.set_label("Tumor perturbation (%)", rotation=270, fontsize=8)
    plt.ylabel("Perturbed \n target(s)")
    if save_fig:
        plt.savefig("multi_strat.svg", format="svg", dpi=300, bbox_inches="tight")

    plt.show()
    return infiltrate_lower, infiltrate_upper


def plot_correlation(sorted_results, save_fig=False):
    # remove _mRNA from the column names if present
    sorted_results["Variable"] = sorted_results["Variable"].str.split("_").str[0]
    fig, ax = plt.subplots(figsize=(16, 6.4))
    plt.bar(
        sorted_results["Variable"], sorted_results["Correlation"], color="lightgray"
    )
    plt.xticks(rotation=50, fontsize=26)
    plt.yticks(fontsize=24)
    ax.yaxis.set_major_formatter(ScalarFormatter(useMathText=True))
    ax.ticklabel_format(axis="y", style="sci", scilimits=(0, 0))

    if save_fig:
        plt.savefig("correlation_barplot.svg", dpi=300, bbox_inches="tight")
    plt.show()


def plot_mutual_info(sorted_results, save_fig=False):
    # remove _mRNA from the column names if present
    sorted_results["Variable"] = sorted_results["Variable"].str.split("_").str[0]
    fig, ax = plt.subplots(figsize=(16, 6.4))
    plt.bar(
        sorted_results["Variable"],
        sorted_results["mutual information"],
        color="lightgray",
    )
    plt.xticks(rotation=50, fontsize=26)
    plt.yticks(fontsize=24)
    ax.yaxis.set_major_formatter(ScalarFormatter(useMathText=True))
    ax.ticklabel_format(axis="y", style="sci", scilimits=(0, 0))

    if save_fig:
        plt.savefig("mutual_info_barplot.svg", dpi=300, bbox_inches="tight")
    plt.show()


def get_example_Tcell_map(dataset, tcell_level, img=23, save_fig=False):

    f = h5py.File(dataset.patch_path, "r")

    # access metadata in the h5 file

    metadata = f["metadata"]

    # convert metadata to a pandas dataframe
    metadata_df = pd.DataFrame(metadata[:])

    # get only rows with tumor
    df = metadata_df[
        (metadata_df["ImageNumber"] == img) & metadata_df["Contains_Tumor"]
    ]
    df = df.merge(tcell_level[["patch_id", "strategy_1"]], on="patch_id", how="left")

    # Determine the size of the grid

    # Initialize the figure with 2 subplots
    _, axes = plt.subplots(nrows=1, ncols=2, figsize=(10, 5))

    # Plot for Contains_Tcytotoxic
    plot_grid(axes[0], df, "Contains_Tcytotoxic", [80, 164, 108], [218, 218, 218])
    axes[0].set_title("Original")

    # Plot for strategy_1
    plot_grid(axes[1], df, "strategy_1", [80, 164, 108], [218, 218, 218])
    axes[1].set_title("Strategy 1")

    if save_fig:
        plt.savefig("strategy_1_example.svg", dpi=300, bbox_inches="tight")

    plt.show()


def make_multiple_Tcell_map(
    dataset, threshold, column=["Contains_Tcytotoxic", "pred_orig"], save_fig=False
):
    # get test data
    X, _, test_metadata, model = get_data_and_model(dataset, "test")

    # get the predictions
    test_metadata["pred_orig"] = model(X) > threshold

    # remove patches with channel values summing to zero
    test_metadata = test_metadata[X.sum(axis=(1, 2, 3)) != 0]

    # select images with at least 300 patches
    df = test_metadata.groupby("ImageNumber").filter(lambda x: len(x) > 150)

    # select images with at least 5 Tcytotoxic patches
    df = df.groupby("ImageNumber").filter(lambda x: x["Contains_Tcytotoxic"].sum() > 5)

    # Initialize the figure with multiple subplots
    # Get unique image numbers
    image_numbers = df["ImageNumber"].unique()
    num_images = len(image_numbers)

    # Define number of rows and columns for subplots
    num_cols = 4
    num_rows = (
        (num_images + num_cols - 1) // num_cols * len(column)
    )  # Ceiling division to ensure all images fit

    # Initialize the figure with multiple subplots
    _, axes = plt.subplots(
        nrows=num_rows, ncols=num_cols, figsize=(5, num_rows * 2.5 / 2)
    )
    axes = axes.flatten()

    for i, img_num in enumerate(image_numbers):
        img_df = df[df["ImageNumber"] == img_num]
        for j, col in enumerate(column):
            plot_grid(
                axes[i * len(column) + j],
                img_df,
                col,
                [25, 152, 79],
                [171, 61, 61],
                gridline=False,
                tumor_only=False,
                background_color=[250, 248, 209],
                max_x=24,
                max_y=24,
                border=True,
            )
        # axes[i].set_title(f"Image Number: {img_num}")
    for ax in axes[(num_images * len(column)) :]:
        ax.axis("off")

    plt.tight_layout()
    if save_fig:
        plt.savefig(f"tcell_map_multiple.svg", dpi=300, bbox_inches="tight")
    plt.show()


# Function to create and plot the grid
def plot_grid(
    ax,
    data,
    color_column,
    true_color,
    false_color,
    gridline=True,
    background_color=255,
    max_x=None,
    max_y=None,
    border=False,
    tumor_only=True,
):
    if max_x is None:
        max_x = data["PatchIndex_X"].max() + 1
    if max_y is None:
        max_y = data["PatchIndex_Y"].max() + 1

    # Create a grid initialized with white (1's multiplied by 255 for RGB)
    grid = np.ones((max_x, max_y, 3), dtype=int) * background_color

    # Apply conditions to assign colors
    for _, row in data.iterrows():
        x = row["PatchIndex_X"]
        y = row["PatchIndex_Y"]
        if tumor_only:
            if row["Contains_Tumor"]:
                if row[color_column]:
                    grid[x, y] = true_color  # True color (green for cytotoxic)
                else:
                    grid[x, y] = false_color  # False color (gray)
        else:
            if row[color_column]:
                grid[x, y] = true_color
            else:
                grid[x, y] = false_color

    # print proportion of tumor patches with Tcytotoxic
    # print(
    #     f"Proportion of tumor patches with Tcytotoxic: {data[data['Contains_Tumor']][color_column].mean()}"
    # )

    # Plotting the grid with visible white grid lines
    ax.imshow(grid, interpolation="nearest", origin="upper", aspect="equal")

    if gridline:
        # Adding white grid lines manually
        for x in range(max_x + 1):
            ax.axhline(x - 0.5, color="white", linestyle="-", linewidth=1.5)
        for y in range(max_y + 1):
            ax.axvline(y - 0.5, color="white", linestyle="-", linewidth=1.5)

    if border:
        for spine in ax.spines.values():
            spine.set_visible(True)
            spine.set_linewidth(0.7)
            spine.set_edgecolor("black")

        # Ensure ticks and labels are not displayed
        ax.set_xticks([])
        ax.set_yticks([])
    else:
        ax.axis("off")
