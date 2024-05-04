import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib import colors
import matplotlib.lines as mlines
import scipy.cluster.hierarchy as sch


def plot_patient_perturbation(
    rel_perturbation: pd.DataFrame,
    channel_to_perturb: list,
    numClust: int = 2,
    extra_cols: list = [],
):
    """
    Plot the perturbation of each patient as well as the median perturbation of each cluster of patients
    """
    # Remove normal tissue samples (for crc dataset only)
    if "type" in rel_perturbation.columns:
        rel_perturbation = rel_perturbation[rel_perturbation["type"] != "Nor"]

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
        method="average" if vmax > 500 else "ward",
        metric="correlation" if vmax > 500 else "euclidean",
        xticklabels=[x[:-5] if x.endswith("_mRNA") else x for x in channel_to_perturb],
        # yticklabels=False,
        norm=colors.TwoSlopeNorm(vmin=vmin, vcenter=0, vmax=vmax / 4 + 1),
        figsize=(8, 8),
    )

    # if FLD or Cancer_Stage is in patient_cf, merge them into grouped_cf using PatientID
    if extra_cols:
        grouped_cf = grouped_cf.join(
            patient_cf[extra_cols],
            on="PatientID",
            how="left",
        )

    # Plot the median perturbation of each cluster of patients
    clusters = sch.fcluster(axe.dendrogram_row.linkage, numClust, "maxclust")

    _, ax = plt.subplots(numClust, 1, figsize=(8, numClust * 1.5))
    strategy_cluster = []
    # add a column to grouped_cf to store the cluster number
    grouped_cf["cluster"] = clusters
    for ii in range(numClust):
        c = np.array(grouped_cf.index[clusters == ii + 1])
        strategy_cluster.append(
            np.median(grouped_cf.loc[c, channel_to_perturb], axis=0)
        )
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

    plt.show()

    strategy_cluster = pd.DataFrame(strategy_cluster, columns=channel_to_perturb)

    if extra_cols:
        print(grouped_cf.groupby("cluster")[extra_cols].value_counts())

    return strategy_cluster, grouped_cf


def aggregate_performance_per_patient(tcell_level, patient_phenotype):
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

    return tcell_level_patient, tcell_level_image


def plot_perturbation_performance(
    tcell_level: pd.DataFrame,
    patient_phenotype: str,
    strategy_mapping: dict,
    strat2_color="#2ba02c",
    strat1_color="#faaf40",
):
    # Aggregate performance per patient
    tcell_level_patient, tcell_level_image = aggregate_performance_per_patient(
        tcell_level, patient_phenotype
    )

    # map each patient to a strategy based on the patient_phenotype and strategy_mapping
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

    if patient_phenotype == "Cancer_Stage":  # melanoma dataset
        plot_horizontal_bar(
            tcell_level_patient,
            patient_phenotype,
            strategy_mapping,
            strat1_color,
            strat2_color,
        )

    elif patient_phenotype == "FLD":  # crc dataset
        plot_two_vertical_bar(
            tcell_level_patient,
            tcell_level_image,
            patient_phenotype,
            strat1_color,
            strat2_color,
        )
    return


def plot_horizontal_bar(
    tcell_level_patient, patient_phenotype, strategy_mapping, strat1_color, strat2_color
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
    plt.show()


def plot_two_vertical_bar(
    tcell_level_patient,
    tcell_level_image,
    patient_phenotype,
    strat1_color,
    strat2_color,
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
    plt.show()


def make_line_plots(
    tcell_level_image, patient_phenotype, strategy_mapping, strat1_color, strat2_color
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
    plt.show()


def plot_umap_embedding(embedding_df, umap_cf):
    cond1 = (embedding_df["Contains_Tcytotoxic"] == 0) & (
        embedding_df["Contains_Tumor"] == 1
    )

    cond2 = (embedding_df["Contains_Tcytotoxic"] == 1) & (
        embedding_df["Contains_Tumor"] == 1
    )

    plt.figure(figsize=(8, 5))

    plt.scatter(
        x=embedding_df.loc[cond1, 'umap1'],
        y=embedding_df.loc[cond1, 'umap2'],
        s=0.2,
        c="#FF0000",
        alpha=0.5,
    )
    plt.scatter(
        x=embedding_df.loc[cond2, 'umap1'],
        y=embedding_df.loc[cond2, 'umap2'],
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
            label="no T cells",
        ),
        mlines.Line2D(
            [0],
            [0],
            color="#04b497",
            marker="o",
            linestyle="None",
            markersize=5,
            label="T cells",
        ),
    ]

    plt.legend(handles=legend_elements)

    plt.axis("off")  # Turn off the axis
    plt.show()

    # group the perturbed points into clusters
    from sklearn.cluster import KMeans

    x = np.array(umap_cf[["perturbed_umap1", "perturbed_umap2"]])
    # x = df_perturbed_normalized.to_numpy()

    # Create a KMeans object with 2 clusters
    ncluster = 9
    kmeans = KMeans(n_clusters=ncluster, random_state=0)

    # Fit the KMeans object to the data
    kmeans.fit(x)

    # Get the labels of the clusters
    cluster_labels = kmeans.labels_
    label_counts = np.bincount(cluster_labels)

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
        "tab:orange",
        "tab:blue",
        "darkgray",
        "darkgray",
        "darkgray",
        "tab:green",
        "tab:purple",
        "darkgray",
        "darkgray",
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
            alpha = 0.8
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
        label="Perturbed",
        alpha=[1 if color_map[ii] != "darkgray" else 0 for ii in cluster_labels],
    )
    plt.axis("off")  # Turn off the axis
    plt.show()
