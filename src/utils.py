from typing import Optional, Tuple, Dict, Any, List, Callable
import os
import warnings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import skimage
import matplotlib
import pickle
from collections import OrderedDict
import copy

from zennit import core as zcore
from zennit import types as ztypes
from zennit import layer as zlayer
from zennit import canonizers as zcanon
from zennit import rules as zrules
from zennit import attribution as zattr
from zennit import composites as zcomp

from metaquantus import (
    load_obj,
    make_benchmarking_df,
    make_benchmarking_df_as_str,
)
from .configs import COLOR_MAP, LABEL_MAP, PLOTTING_COLOURS


def get_indices(batch_size: int = 25, dataset_length: int = 300):
    """
    Generate a list of index ranges for splitting a dataset into batches.

    Parameters:
    batch_size (int, optional)
        The size of each batch. Default is 25.
    dataset_length (int, optional)
        The total length of the dataset. Default is 300.

    Returns:
    list: A list of index ranges, each specified as [start, end], for splitting the dataset into batches.
    """
    return [
        [start, start + batch_size]
        for start in range(0, batch_size * (dataset_length // batch_size), batch_size)
    ]


def create_feature_mask(input_tensor: torch.Tensor, nr_segments: int) -> torch.Tensor:
    """
    Create a feature mask for an input tensor using SLIC segmentation.

    Parameters
    ----------
    input_tensor : torch.Tensor
        The input tensor of shape (C, H, W) where C is the number of channels, and H and W are the height and width.

    nr_segments : int
        The number of segments to generate using SLIC segmentation.

    Returns
    -------
    torch.Tensor
        A tensor containing segment labels of the same shape as the input tensor.

    Notes
    -----
    - If the input tensor is a single-channel image, it will be converted to an RGB image before applying SLIC.
    - The SLIC algorithm is used for superpixel segmentation.

    """
    C, H, W = input_tensor.shape
    img_np = input_tensor.permute(1, 2, 0).cpu().numpy()

    if C == 1:
        img_np = skimage.color.gray2rgb(img_np)

    segments = skimage.segmentation.slic(img_np, n_segments=nr_segments, compactness=10)
    segments_tensor = torch.from_numpy(segments).squeeze()
    return segments_tensor


def create_masks_for_dataset(
    dataset: torch.Tensor, nr_segments: int = 50
) -> torch.Tensor:
    """
    Create feature masks for a dataset of input tensors.

    Parameters
    ----------
    dataset : torch.Tensor
        The dataset of input tensors with shape (n, C, H, W), where n is the number of samples,
        C is the number of channels, and H and W are the height and width.

    nr_segments : int, optional
        The number of segments to generate for each input tensor using SLIC segmentation.

    Returns
    -------
    torch.Tensor
        A tensor containing feature masks for each input tensor in the dataset.
        The shape of the output tensor is (n, H, W).

    Notes
    -----
    - This function applies the `create_feature_mask` function to each input tensor in the dataset.

    """
    n, C, H, W = dataset.shape
    masks = torch.empty((n, H, W), dtype=torch.int)

    for sample_id in range(n):
        masks[sample_id] = create_feature_mask(
            input_tensor=dataset[sample_id], nr_segments=nr_segments
        )

    return masks


def replace_none_with_nan(x: any) -> float:
    """
    Replace None values with NaN.

    Parameters
    ----------
    x : any
        The input value.

    Returns
    -------
    float
        The input value if it's not None, otherwise NaN.
    """

    return x if x is not None else np.nan


def count_files(path):
    """
    Count the number of files in a directory.

    Parameters
    ----------
    path: str
        The path to the directory.

    Returns
    -------
    int
        The number of files in the directory.
    """
    return len([f for f in os.listdir(path) if os.path.isfile(os.path.join(path, f))])


def plot_randomisation_curves(
    scores: Dict[str, dict],
    dataset_name: str,
    model_name: str,
    metric_name: str = "eMPRT",
    title_str: Optional[str] = None,
    figsize: Tuple[int, int] = (8, 4),
):
    """
    Plot eMPRT curves for different methods.

    Parameters
    ----------
    scores : dict
        A dictionary containing method names as keys and score data as values.
        Example: {"method1": {"explanation_scores": {...}}, "method2": {"explanation_scores": {...}}}

    dataset_name : str
        The name of the dataset.

    model_name : str
        The name of the model.

    metric_name: str, optional
        The name of the metric. Default is "eMPRT".

    title_str : str, optional
        Additional title information.

    figsize : Tuple[int, int], optional
        Figure size (width, height).

    Returns
    -------
    None
    """
    fig, ax = plt.subplots(1, 1, figsize=figsize)

    for ix, (method, data) in enumerate(scores.items()):
        if "explanation_scores" in data:  # This is Efficient MPRT.
            means = []
            stds = []
            model_means = []
            model_stds = []
            layer_names = []

            for (layer, values), (_, model_values) in zip(
                data["explanation_scores"].items(),
                data["model_scores"].items(),
            ):
                layer_names.append(layer)
                values = np.vectorize(replace_none_with_nan)(values)
                model_values = np.vectorize(replace_none_with_nan)(model_values)
                means.append(np.nanmean(values))
                stds.append(np.nanstd(values))
                model_means.append(np.nanmean(model_values))
                model_stds.append(np.nanstd(model_values))

            if ix == 0:
                plt.plot(
                    layer_names,
                    model_means,
                    "o-",
                    color="black",
                    label=f"Model",
                )
                plt.fill_between(
                    layer_names,
                    np.array(model_means) + np.array(model_stds),
                    np.array(model_means) - np.array(model_stds),
                    color="black",
                    alpha=0.2,
                )

            try:
                emprt_score = np.nanmean(data["evaluation_scores"])
            except:
                emprt_score = np.nanmean(data["rate_of_change_scores"])

            plt.plot(
                layer_names,
                means,
                "o-",
                color=COLOR_MAP[method],
                label=LABEL_MAP[method]
                + f" (eMPRT={emprt_score:.2f},"
                + r" $\rho$"
                + f"={np.nanmean(data['correlation_scores']):.2f})",
            )
            plt.fill_between(
                layer_names,
                np.array(means) + np.array(stds),
                np.array(means) - np.array(stds),
                color=COLOR_MAP[method],
                alpha=0.2,
            )
        elif (
            "similarity_scores" in data
        ):  # This is MPRT and Smooth MPRT. #scores_evaluation
            means = []
            stds = []
            layer_names = []
            for layer, values in data["similarity_scores"].items():
                layer_names.append(layer)
                values = np.vectorize(replace_none_with_nan)(values)
                means.append(np.nanmean(values))
                stds.append(np.nanstd(values))

            plt.plot(
                layer_names,
                means,
                "o-",
                color=COLOR_MAP[method],
                label=LABEL_MAP[method] + f" SSIM={np.nanmean(means):.2f})",
            )
            plt.fill_between(
                layer_names,
                np.array(means) + np.array(stds),
                np.array(means) - np.array(stds),
                color=COLOR_MAP[method],
                alpha=0.2,
            )

    ax.set_xlabel("Layers")
    if model_name == "ResNet18":
        ax.set_xticks(list(range(len(layer_names)))[::4])
        xticklabels = layer_names[::4]
        ax.set_xticklabels(layer_names[::4], rotation=45)
    elif model_name == "VGG16":
        ax.set_xticks(list(range(len(layer_names)))[::2])
        xticklabels = layer_names[::2]
    else:
        ax.set_xticks(list(range(len(layer_names))))
        ax.set_xticklabels(layer_names, rotation=45)
        xticklabels = layer_names
    for i in range(len(xticklabels)):
        xticklabels[i] = xticklabels[i].replace("downsample", "ds")
    xticklabels = xticklabels[:-1] + ["final"]
    ax.set_xticklabels(xticklabels, rotation=45)

    if metric_name == "eMPRT":
        plt.ylabel("Complexity $H(\\cdot)$")
    else:  # This is MPRT and Smooth MPRT.
        plt.ylabel("SSIM")
    title = f"{metric_name}\n({model_name} {dataset_name})"

    if title_str is not None:
        title += f"\n{title_str}"

    plt.title(title)
    plt.legend(loc="center left", bbox_to_anchor=(1, 0.5), ncol=1)
    plt.grid(True)


def concat_imagenet_benchmarks(
    benchmarks: dict, estimators: dict, category: str = "Randomisation"
) -> pd.DataFrame:
    """
    Concat different benchmark results from batch job on ImageNet data.

    Parameters
    ----------
    benchmark: dict
        The benchmarking data.
    estimators: dict
        The estimators used in the experiment.
    category: str
        The evaluation category.
    Returns
    -------
        pd.DataFrame
    """

    dfs = []
    for batch, benchmark in benchmarks.items():
        df = make_benchmarking_df(benchmark=benchmark, estimators=estimators)
        dfs.append(df)
    df = pd.concat(dfs)
    columns_to_exclude = ["Category", "Estimator", "Test"]
    df.drop(columns="Category", inplace=True)
    for col in df.columns:
        if col not in columns_to_exclude:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    df_imagenet = df.groupby(["Estimator", "Test"]).mean()
    df_imagenet.reset_index(inplace=True)
    df_imagenet["Category"] = category
    return df_imagenet


def make_benchmarking_df_imagenet_as_str(df_imagenet: pd.DataFrame, estimators: Dict):
    """
    Create the benchmarking df.

    Parameters
    ----------
    benchmark: dict
        The benchmarking data.
    estimators: dict
        The estimators used in the experiment.

    Returns
    -------
    df
    """
    df = pd.DataFrame(
        columns=[
            "Category",
            "Estimator",
            "Test",
            "MC_bar",
            "MC",
            "IAC_{NR}",
            "IAC_{AR}",
            "IEC_{NR}",
            "IEC_{AR}",
        ]
    )

    mc_bars = (
        df_imagenet.groupby(["Estimator"])[["MC", "MC std"]]
        .mean()
        .reset_index(level=["Estimator"])
    )

    scores = ["IAC_{NR}", "IAC_{AR}", "IEC_{NR}", "IEC_{AR}"]
    row = 0
    for ex1, (estimator_category, metrics) in enumerate(estimators.items()):
        for ex2, estimator_name in enumerate(metrics):
            for px, perturbation_type in enumerate(["Model", "Input"]):
                row += ex1 + ex2 + px
                df.loc[row, "Test"] = perturbation_type
                if px == 1:
                    df.loc[row, "Category"] = estimator_category
                    df.loc[row, "Estimator"] = estimator_name
                else:
                    df.loc[row, "Category"] = estimator_category
                    df.loc[row, "Estimator"] = estimator_name

                mc_bar_mean = mc_bars.loc[
                    (mc_bars["Estimator"] == estimator_name), "MC"
                ].to_numpy()[0]
                mc_bar_std = mc_bars.loc[
                    (mc_bars["Estimator"] == estimator_name), "MC std"
                ].to_numpy()[0]
                if perturbation_type == "Model":
                    df.loc[row, "MC_bar"] = (
                        f"{mc_bar_mean:.2f}" + " $\pm$ " + f"{mc_bar_std * 2:.2f}"
                    ) + " &"
                else:
                    df.loc[row, "MC_bar"] = ""
                mc = df_imagenet.loc[
                    (df_imagenet["Estimator"] == estimator_name)
                    & (df_imagenet["Test"] == perturbation_type),
                    "MC",
                ].to_numpy()[0]
                mc_std = df_imagenet.loc[
                    (df_imagenet["Estimator"] == estimator_name)
                    & (df_imagenet["Test"] == perturbation_type),
                    "MC std",
                ].to_numpy()[0]
                if perturbation_type == "Input":
                    df.loc[row, "MC"] = (
                        "\CC{30}{"
                        + (f"{mc:.2f}" + " $\pm$ " + f"{mc_std * 2:.2f}")
                        + "} &"
                    )
                else:
                    df.loc[row, "MC"] = (
                        f"{mc:.2f}" + " $\pm$ " + f"{mc_std * 2:.2f}"
                    ) + " &"

                for s in scores:

                    score_mean = df_imagenet.loc[
                        (df_imagenet["Estimator"] == estimator_name)
                        & (df_imagenet["Test"] == perturbation_type),
                        s,
                    ].to_numpy()[0]
                    score_std = df_imagenet.loc[
                        (df_imagenet["Estimator"] == estimator_name)
                        & (df_imagenet["Test"] == perturbation_type),
                        s + " std",
                    ].to_numpy()[0]
                    if perturbation_type == "Input":
                        df.loc[row, s] = (
                            "\CC{30}{"
                            + f"{score_mean:.2f}"
                            + " $\pm$ "
                            + f"{score_std * 2:.2f}"
                            + "} &"
                        )
                    else:
                        df.loc[row, s] = (
                            f"{score_mean:.2f}"
                            + " $\pm$ "
                            + f"{score_std * 2:.2f}"
                            + " &"
                        )

    return df


def prepare_scores_for_area_plots(
    benchmarks: Dict[str, Dict[str, Any]],
    metrics: List[str],
    category: str,
    path_results: str,
) -> Dict[str, Dict[str, Any]]:
    """
    Create a dictionary containing scores for each benchmark.

    Parameters
    ----------
    benchmark: dict
        The benchmarking data.
    metrics: list
        A list of strings of the metrics.
    category: str
        The evaluation category.
    path_results: str
        The path to the results.

    Returns
    -------
        dict
    """
    scores_meta = {}
    for bx1, (benchmark_meta, benchmark_curr) in enumerate(benchmarks.items()):
        dataset_name, model_name, xai_subset = benchmark_meta.split("_")
        scores_meta[benchmark_meta] = {}

        for ex1, estimator_name in enumerate(metrics):

            mc_scores = []
            mc_stds = []
            batches = 1
            try:
                benchmark = benchmark_curr[category][estimator_name]
            except:
                batches = len(benchmark_curr)

            scores = {}
            scores_meta[benchmark_meta][estimator_name] = {}

            # Collect scores over perturbation types.
            for px, perturbation_type in enumerate(["Input", "Model"]):

                scores[perturbation_type] = {
                    "IAC_NR": [],
                    "IAC_AR": [],
                    "IEC_NR": [],
                    "IEC_AR": [],
                }

                for i in benchmark_curr.keys():

                    try:
                        benchmark = benchmark_curr[i][category][estimator_name]
                    except:
                        pass
                    scores[perturbation_type]["IAC_NR"].append(
                        np.array(
                            benchmark["results_consistency_scores"][perturbation_type][
                                "intra_scores_res"
                            ]
                        )
                    )
                    scores[perturbation_type]["IAC_AR"].append(
                        np.array(
                            benchmark["results_consistency_scores"][perturbation_type][
                                "intra_scores_adv"
                            ]
                        )
                    )
                    scores[perturbation_type]["IEC_NR"].append(
                        np.array(
                            benchmark["results_consistency_scores"][perturbation_type][
                                "inter_scores_res"
                            ]
                        )
                    )
                    scores[perturbation_type]["IEC_AR"].append(
                        np.array(
                            benchmark["results_consistency_scores"][perturbation_type][
                                "inter_scores_adv"
                            ]
                        )
                    )

                    mc_scores.append(
                        benchmark["results_meta_consistency_scores"][perturbation_type][
                            "MC_mean"
                        ]
                    )
                    mc_stds.append(
                        benchmark["results_meta_consistency_scores"][perturbation_type][
                            "MC_std"
                        ]
                    )
                    # print(benchmark["results_meta_consistency_scores"])

            # Append scores.
            scores_meta[benchmark_meta][estimator_name]["scores"] = scores
            scores_meta[benchmark_meta][estimator_name]["mc_scores"] = mc_scores
            scores_meta[benchmark_meta][estimator_name]["mc_stds"] = mc_stds

            # Flatten all scores within the perturbation type.
            for px, perturbation_type in enumerate(["Input", "Model"]):
                for criterion in scores[perturbation_type].keys():
                    scores[perturbation_type][criterion] = np.array(
                        scores[perturbation_type][criterion]
                    ).flatten()

    # Save data.
    with open(path_results + f"meta_evaluation_scores.pickle", "wb") as f:
        pickle.dump(scores_meta, f)

    return scores_meta


def plot_area_graph(
    scores_meta: Dict[str, Dict[str, Any]],
    benchmarks: Dict[str, Dict[str, Any]],
    metrics: List[str],
    category: str,
    colours: Dict[str, str],
    figsize: Tuple[int, int] = (16, 8),
    save: bool = False,
    path: Optional[str] = None,
    fname: Optional[str] = None,
    **kwargs: Any,
) -> None:
    """

    Parameters
    ----------
    scores_meta: dict
        The scores used for plotting.
    benchmarks: dict
        The benchmarks used for plotting.
    metrics: list
        A list of strings of the metrics
    category: str
        The evaluation category.
    colours: Dict[str, str]
        A dictionary containing colours for each estimator.
    figsize: Tuple[int, int]
        The figure size.
    save: bool
        Whether to save the figure.
    path: str
        The path to save the figure.
    fname: str
        The filename.
    kwargs: Any
        Additional keyword arguments.

    Returns
    -------

    """
    n_rows = kwargs.get("n_rows", len(metrics))
    n_cols = kwargs.get("n_cols", len(benchmarks))

    fig, axs = plt.subplots(n_rows, n_cols, sharex=True, figsize=figsize)

    for bx1, (benchmark_meta, benchmark_curr) in enumerate(benchmarks.items()):
        dataset_name, model_name, xai_subset = benchmark_meta.split("_")
        for ex1, estimator_name in enumerate(metrics):

            scores = scores_meta[benchmark_meta][estimator_name]["scores"]
            mc_scores = scores_meta[benchmark_meta][estimator_name]["mc_scores"]

            # Settings.
            colour_setting = f"{estimator_name}_{dataset_name}_{model_name}"
            hatches = {"Input": "", "Model": "/"}

            # Set values for m* and the actual values by the estimator.
            X_gt = [-1, 0, 1, 0]
            Y_gt = [0, 1, 0, -1]

            # Plot m*.
            axs[ex1, bx1].fill(X_gt, Y_gt, color="black", alpha=0.075, label="m*")

            # Fill area.
            for px, perturbation_type in enumerate(["Input", "Model"]):
                X_area = [
                    -scores[perturbation_type]["IAC_AR"].mean(),
                    0,
                    scores[perturbation_type]["IEC_AR"].mean(),
                    0,
                ]
                Y_area = [
                    0,
                    scores[perturbation_type]["IAC_NR"].mean(),
                    0,
                    -scores[perturbation_type]["IEC_NR"].mean(),
                ]
                axs[ex1, bx1].fill(
                    X_area,
                    Y_area,
                    color=PLOTTING_COLOURS[colour_setting],
                    alpha=0.75,
                    label=perturbation_type,
                    edgecolor="black",
                    hatch=hatches[perturbation_type],
                )

            # Annotate the labels.
            axs[ex1, bx1].annotate("${IAC}_{AR}$", (-1, 0), fontsize=12)
            axs[ex1, bx1].annotate("${IAC}_{NR}$", (-0.2, 0.8), fontsize=12)
            axs[ex1, bx1].annotate("${IEC}_{AR}$", (0.7, 0), fontsize=12)
            axs[ex1, bx1].annotate("${IEC}_{NR}$", (-0.2, -0.8), fontsize=12)

            # Title and grids.
            axs[ex1, bx1].set_title(
                # f"{estimator_name_plotting} ({np.array(mc_scores).flatten().mean():.4f})",
                f"{estimator_name} ({np.mean(mc_scores):.3f})",  # ({np.mean(mc_stds):.3f})
                fontsize=15,
            )
            axs[ex1, bx1].legend(fontsize=13, loc="upper left")

            # Labels.
            tick_locations = [1.0, 0.5, 0.0, -0.5, -1.0]
            tick_labels = ["1.0", "0.5", "0.0", "0.5", "1.0"]

            # If last row.
            axs[ex1, bx1].set_xticks(tick_locations)
            if ex1 == len(metrics) - 1:
                axs[ex1, bx1].set_xticklabels(tick_labels, fontsize=14)
            else:
                axs[ex1, bx1].set_xticklabels([])

            # If first column.
            axs[ex1, bx1].set_yticks(tick_locations)
            if bx1 == 0:
                axs[ex1, bx1].set_yticklabels(tick_labels, fontsize=14)
            else:
                axs[ex1, bx1].set_yticklabels([])

            axs[ex1, bx1].grid()

            # If last row.
            if ex1 == len(metrics) - 1:
                axs[ex1, bx1].set_xlabel(
                    f"{dataset_name}, {model_name}", fontsize=14, weight="bold"
                )

    plt.tight_layout()
    if save:
        plt.savefig(
            path + f"plots/full_area_graph_{category}_{fname}.svg",
            # dpi=500,
        )
        plt.savefig(
            path + f"plots/full_area_graph_{category}_{fname}.pdf",
            # dpi=500,
        )


def prepare_and_plot_area_graph(
    benchmarks: dict,
    xai_set: str,
    path_results: str,
    suffix: str = "",
):
    """

    Parameters
    ----------
    benchmarks: dict
    xai_set: str
        The name of the xai_group.
    suffix: str
        A suffix for naming.

    Returns
    -------

    """

    metrics = ["eMPRT", "sMPRT", "MPRT"]
    scores_area_plot = prepare_scores_for_area_plots(
        benchmarks=benchmarks,
        metrics=metrics,
        category="Randomisation",
        path_results=path_results,
    )

    means = []
    stds = []
    for dataset in scores_area_plot.keys():
        for metric in metrics:
            mc_scores = np.array(scores_area_plot[dataset][metric]["mc_scores"])
            mean = mc_scores.mean()
            std = mc_scores.std()
            # print(f"{dataset} - {metric} - {mean} $\pm$ {std} & ")
            means.append(mean)
            stds.append(std)

    with warnings.catch_warnings():
        plot_area_graph(
            scores_meta=scores_area_plot,
            benchmarks=benchmarks,
            metrics=metrics,
            category="Randomisation",
            colours=PLOTTING_COLOURS,
            figsize=(len(benchmarks) * 3, len(benchmarks) * 2.5),
            save=True,
            path=path_results,
            fname=xai_set + suffix,
        )


def get_results_bar(
    all_benchmarks: dict, path_results: str
) -> Dict[str, Dict[str, Any]]:
    """
    Get results for bar chart.

    Parameters
    ----------
    all_benchmarks: dict
        A dict of dicts with all benchmark data across tasks.
    path_results: str
        The path to results.

    Returns
    -------
        dict
    """

    metrics = ["eMPRT", "sMPRT", "MPRT"]

    # Compute means of scores.
    results_bar = {}
    means = []
    stds = []
    scores_meta = prepare_scores_for_area_plots(
        benchmarks=all_benchmarks,
        metrics=metrics,
        category="Randomisation",
        path_results=path_results,
    )

    for meta_data in scores_meta.keys():
        dataset_name, model_name, xai_subset = meta_data.split("_")

        if meta_data not in results_bar:
            results_bar[meta_data] = {}
        results_bar[meta_data] = {}

        for metric in metrics:

            mc_scores = np.array(scores_meta[meta_data][metric]["mc_scores"])
            mean = round(mc_scores.mean(), 3)
            std = round(mc_scores.std(), 3)
            # print(f"{meta_data} - {metric} - {mean} $\pm$ {std} & ")
            results_bar[meta_data][metric] = f"{mean} $\pm$ {std}"
            means.append(mean)
            stds.append(std)

    sorted_keys = [
        "MNIST_LeNet_GSxIG",
        "MNIST_LeNet_SAxLRPplusxIXG",
        "MNIST_LeNet_GxGCxLRPepsxGB",
        "MNIST_LeNet_GPxGSxGCxLRPepsxSA",
        "fMNIST_LeNet_GSxIG",
        "fMNIST_LeNet_SAxLRPplusxIXG",
        "fMNIST_LeNet_GxGCxLRPepsxGB",
        "fMNIST_LeNet_GPxGSxGCxLRPepsxSA",
        "ImageNet_ResNet18_GSxIG",
        "ImageNet_ResNet18_SAxLRPplusxIXG",
        "ImageNet_ResNet18_GxGCxLRPepsxGB",
        "ImageNet_ResNet18_GPxGSxGCxLRPepsxSA",
        "ImageNet_VGG16_GSxIG",
        "ImageNet_VGG16_SAxLRPplusxIXG",
        "ImageNet_VGG16_GxGCxLRPepsxGB",
        "ImageNet_VGG16_GPxGSxGCxLRPepsxSA",
    ]
    results_bar = OrderedDict((k, results_bar[k]) for k in sorted_keys)

    return results_bar


def plot_hierarchical_bar(
    data: dict,
    metrics: List[str],
    path_results: str,
    suffix: str = "",
    ystart: float = 0.45,
    yend: float = 0.95,
) -> None:
    """
    Plot hierarchical bar chart.

    Parameters
    ----------
    data: dict
        The benchmarking data for the plotting.
    metrics: List[str]
        The metrics to plot.
    path_results: str
        The path for the results.
    suffix: str
        The suffix for the plot.
    ystart: float
        The ylim start.
    yend: float
        The ylim end.

    Returns
    -------
    None
    """

    datasets = []
    xai_groups = []
    for ix, meta_name in enumerate(data.keys()):
        dataset_name, model_name, xai_group = meta_name.split("_")
        if dataset_name + "_" + model_name not in datasets:
            datasets.append(dataset_name + "_" + model_name)
        if xai_group not in xai_groups:
            xai_groups.append(xai_group)

    data_keys = list(data.keys())

    colours = {
        "MPRT_MNIST_LeNet": "#d9f0d3",
        "MPRT_fMNIST_LeNet": "#a6dba0",
        "MPRT_ImageNet_VGG16": "#5aae61",
        "MPRT_ImageNet_ResNet18": "#1b7837",
        "eMPRT_MNIST_LeNet": "#d8daeb",
        "eMPRT_fMNIST_LeNet": "#b2abd2",
        "eMPRT_ImageNet_VGG16": "#8073ac",
        "eMPRT_ImageNet_ResNet18": "#542788",
        "sMPRT_MNIST_LeNet": "#fee0b6",
        "sMPRT_fMNIST_LeNet": "#fdb863",
        "sMPRT_ImageNet_VGG16": "#e08214",
        "sMPRT_ImageNet_ResNet18": "#b35806",
    }

    n_datasets = len(datasets)
    n_xai_groups = len(xai_groups)
    n_metrics = len(metrics)

    hatches = [".", "*", "//", ""]
    hatches_styles = np.tile(np.repeat(hatches, n_metrics), n_datasets)

    width = 0.95
    x = np.arange(int(n_datasets * n_xai_groups * n_metrics))

    fig, ax = plt.subplots(figsize=(15, 3.5))

    extra_space = 1.5
    position = 0

    tick_positions = []  # To store the tick positions for big groups

    for i in x:
        if i % (n_xai_groups * n_metrics) == 0 and i != 0:
            # Extra space between different datasets.
            position += 2
        elif i % n_metrics == 0 and i != 0:
            # Almost no space within the dataset.
            position += 0.5

        # Start position for the big group.
        if i % (n_xai_groups * n_metrics) == 0:
            start_position = position

        # Get the choice colour.
        metric_type = metrics[i % n_metrics]
        choice = [
            c
            for c in colours.keys()
            if metric_type in c and datasets[int(i // (n_datasets * n_metrics))] in c
        ][0]

        # Get the data.
        mean, std = data[data_keys[int(i // (n_metrics))]][metric_type].split(
            " $\\pm$ "
        )
        # print(data_keys[int(i//(n_metrics))], metric_type, mean, std)

        # Plot!
        ax.bar(
            position,
            float(mean),
            yerr=float(std),
            width=width,
            color=colours[choice],
            hatch=hatches_styles[i],
            label=metric_type if i < n_metrics else "",
            edgecolor="black",
        )
        position += 1

        if (i + 1) % (n_xai_groups * n_metrics) == 0:
            end_position = position
            tick_positions.append((start_position + end_position - 3) / 2)

    # Details.
    ax.set_xticks(tick_positions)
    labels = [f'{c.split("_")[0]}\n{c.split("_")[1]}' for c in datasets]
    ax.set_xticklabels(labels)
    yticks = np.arange(ystart, yend, 0.05)
    labels = [
        f"{round(i, 2)}" if ix % 4 == 0 and ix != 0 else ""
        for ix, i in enumerate(yticks)
    ]
    ax.set_yticks(labels=labels, ticks=yticks)
    ax.set_ylabel("$\overline{MC}$")
    ax.set_ylim(ystart, yend)

    # Custom legend elements for colours.
    legend_elements_color = [
        matplotlib.patches.Patch(
            facecolor=colours[choice], edgecolor="black", label=metric_type
        )
        for metric_type, choice in zip(
            ["eMPRT", "sMPRT", "MPRT"],
            ["eMPRT_ImageNet_VGG16", "sMPRT_ImageNet_VGG16", "MPRT_ImageNet_VGG16"],
        )
    ]

    # Custom legend elements for hatches.
    legend_elements_hatch = [
        matplotlib.patches.Patch(
            facecolor="white", edgecolor="black", hatch=hatch_style, label=f"$M_{i+2}$"
        )  # _{i+1}
        for i, hatch_style in enumerate(hatches)
    ]

    # Combine all three types of legends: colour (metrics), hatch (XAI method groups).
    legend_elements = legend_elements_color + legend_elements_hatch

    # Create single legend for both.
    ax.legend(handles=legend_elements, loc="upper left", bbox_to_anchor=(1, 1))
    plt.tight_layout(rect=[0, 0, 0.85, 1])
    plt.grid(True)
    plt.savefig(
        path_results + f"plots/mc_scores_datasets{suffix}.svg", bbox_inches="tight"
    )
    plt.savefig(
        path_results + f"plots/mc_scores_datasets{suffix}.pdf", bbox_inches="tight"
    )
    plt.show()


def prepare_benchmarking_results(
    path_results: str, xai_round: str, setup_estimators: Callable
):
    """
    Prepare benchmarking results.

    Parameters
    ----------
    path_results: str
        The path to results.
    xai_round: str
        The XAI group used for benchmarking.
    setup_estimators: Callabe
        The callable for setting up the estimators.
    Returns
    -------

    """

    task = "ImageNet_VGG16"
    dataset_name, model_name = task.split("_")
    path_task = path_results + "benchmarking/" + task + "/" + xai_round + "/"
    benchmark_imagenet_vgg16 = {
        i
        + 1: load_obj(path=path_task, fname=f"batch_{i + 1}", use_json=True)[
            dataset_name
        ][model_name]
        for i in range(count_files(path_task))
    }
    df_imagenet_vgg16 = concat_imagenet_benchmarks(
        benchmarks=benchmark_imagenet_vgg16, estimators=setup_estimators()
    )
    df_imagenet_vgg16_str = make_benchmarking_df_imagenet_as_str(
        df_imagenet=df_imagenet_vgg16, estimators=setup_estimators()
    )

    task = "ImageNet_ResNet18"
    dataset_name, model_name = task.split("_")
    path_task = path_results + "benchmarking/" + task + "/" + xai_round + "/"
    benchmark_imagenet_resnet = {
        i
        + 1: load_obj(path=path_task, fname=f"batch_{i + 1}", use_json=True)[
            dataset_name
        ][model_name]
        for i in range(count_files(path_task))
    }
    df_imagenet_resenet = concat_imagenet_benchmarks(
        benchmarks=benchmark_imagenet_resnet, estimators=setup_estimators()
    )
    df_imagenet_resenet_str = make_benchmarking_df_imagenet_as_str(
        df_imagenet=df_imagenet_resenet, estimators=setup_estimators()
    )

    task = "MNIST_LeNet"
    dataset_name, model_name = task.split("_")
    path_task = path_results + "benchmarking/" + task + "/" + xai_round + "/"
    benchmark_mnist = load_obj(path_task, fname=f"batch_1", use_json=True)[
        dataset_name
    ][model_name]
    df_mnist = make_benchmarking_df(
        benchmark=benchmark_mnist, estimators=setup_estimators()
    )
    df_mnist_str = make_benchmarking_df_as_str(
        benchmark=benchmark_mnist, estimators=setup_estimators()
    )

    task = "fMNIST_LeNet"
    dataset_name, model_name = task.split("_")
    path_task = path_results + "benchmarking/" + task + "/" + xai_round + "/"
    benchmark_fmnist = load_obj(path_task, fname=f"batch_1", use_json=True)[
        dataset_name
    ][model_name]
    df_fmnist = make_benchmarking_df(
        benchmark=benchmark_fmnist, estimators=setup_estimators()
    )
    df_fmnist_str = make_benchmarking_df_as_str(
        benchmark=benchmark_fmnist, estimators=setup_estimators()
    )
    xai_set = xai_round.replace("_", "x")  # for plotting purposes.
    if xai_round == "GP_GS_GC_LRP-Eps_SA":
        xai_set = "GPxGSxGCxLRPepsxSA"
    benchmarks_m1 = {
        f"MNIST_LeNet_{xai_set}": benchmark_mnist,
        f"fMNIST_LeNet_{xai_set}": benchmark_fmnist,
        f"ImageNet_ResNet18_{xai_set}": benchmark_imagenet_resnet,
        f"ImageNet_VGG16_{xai_set}": benchmark_imagenet_vgg16,
    }

    return benchmarks_m1, xai_set

def get_layer(model, layer_name):
    for name, mod in model.named_modules():
        if name == layer_name:
            return mod
        
def randomise_model_layers(model, layer_name, order="top_down"):
    randomised_model = copy.deepcopy(model)

    modules = [
        l
        for l in randomised_model.named_modules()
        if (hasattr(l[1], "reset_parameters"))
    ]

    if order == "target_layer":
        for module in modules:
            if layer_name == module[0]:
                module[1].reset_parameters()
                return randomised_model
        raise ValueError(f"{layer_name} does not exist in model")

    if order == "top_down":
        modules = modules[::-1]

    for module in modules:
        print(order, module[0])
        if layer_name == module[0]:
            break
        module[1].reset_parameters()

    return randomised_model

def store_output_hook(module, input, output):
    module.tmp_stored_output = output

def explain_activation(
    model,
    inputs,
    layer_name,
    device,
    **kwargs,
) -> np.ndarray:

    # Get zennit composite, canonizer, attributor and handle canonizer kwargs.
    canonizer = kwargs.get("canonizer", None)
    if not canonizer == None and not issubclass(canonizer, zcanon.Canonizer):
        raise ValueError(
            "The specified canonizer is not valid. "
            "Please provide None or an instance of zennit.canonizers.Canonizer"
        )

    # Handle attributor kwargs.
    composite = kwargs.get("composite", None)
    if not composite == None and isinstance(composite, str):
        if composite not in zcomp.COMPOSITES.keys():
            raise ValueError(
                "Composite {} does not exist in zennit."
                "Please provide None, a subclass of zennit.core.Composite, or one of {}".format(
                    composite, zcomp.COMPOSITES.keys()
                )
            )
        else:
            composite = zcomp.COMPOSITES[composite]
    if not composite == None and not issubclass(composite, zcore.Composite):
        raise ValueError(
            "The specified composite {} is not valid. "
            "Please provide None, a subclass of zennit.core.Composite, or one of {}".format(
                composite, zcomp.COMPOSITES.keys()
            )
        )

    # Set model in evaluate mode.
    model.eval()

    if not isinstance(inputs, torch.Tensor):
        inputs = torch.Tensor(inputs).to(device)

    canonizer_kwargs = kwargs.get("canonizer_kwargs", {})
    composite_kwargs = kwargs.get("composite_kwargs", {})

    # Initialize canonizer, composite, and attributor.
    if canonizer is not None:
        canonizers = [canonizer(**canonizer_kwargs)]
    else:
        canonizers = []
    if composite is not None:
        composite = composite(
            **{
                **composite_kwargs,
                "canonizers": canonizers,
            }
        )

    layer = get_layer(model, layer_name)
    handle = layer.register_forward_hook(store_output_hook)

    # Get the attributions.
    
    composite.register(model)

    if not inputs.requires_grad:
        inputs.requires_grad = True
    model(inputs)
    activation = layer.tmp_stored_output
    initital_relevance = torch.ones_like(activation)

    explanation, = torch.autograd.grad(
        (activation,),
        (inputs,),
        grad_outputs=(initital_relevance,)
    )

    if isinstance(explanation, torch.Tensor):
        if explanation.requires_grad:
            explanation = explanation.cpu().detach().numpy()
        else:
            explanation = explanation.cpu().numpy()

    # Sum over the axes.
    explanation = np.sum(explanation, axis=1, keepdims=True)

    del layer.tmp_stored_output
    composite.remove()
    handle.remove()

    return explanation

def labelsorter(label, noisedraw_vals=None):

    if noisedraw_vals is not None:
        max_noisedraw_val = np.amax(noisedraw_vals)

        for m, method in enumerate(COLOR_MAP.keys()):
            if method in label:
                for noisedraw_val in noisedraw_vals:
                    if str(noisedraw_val) in label:
                        return max_noisedraw_val * m + noisedraw_val
    else:
        for m, method in enumerate(COLOR_MAP.keys()):
            if method in label:
                return m

    for m, method in enumerate(COLOR_MAP.keys()):
            if method in label:
                return m

def get_random_layer_generator(model, order: str = "top_down"):
    """
    In every iteration yields a copy of the model with one additional layer's parameters randomized.
    For cascading randomization, set order (str) to 'top_down'. For independent randomization,
    set it to 'independent'. For bottom-up order, set it to 'bottom_up'.

    Parameters
    ----------
    order: string
        The various ways that a model's weights of a layer can be randomised.

    Returns
    -------
    layer.name, random_layer_model: string, torch.nn
        The layer name and the model.
    """
    original_parameters = model.state_dict()
    random_layer_model = copy.deepcopy(model)

    modules = [
        l
        for l in random_layer_model.named_modules()
        if (hasattr(l[1], "reset_parameters"))
    ]

    if order == "top_down":
        modules = modules[::-1]

    for module in modules:
        if order == "independent":
            random_layer_model.load_state_dict(original_parameters)
        module[1].reset_parameters()
        yield module[0], random_layer_model

def eval_accuracy(model, loader, device):
    """
    Evaluates Model.
    """

    # Initialize running measures
    n_correct = 0
    n_predicted = 0
    total_labels = []
    total_predictions = []

    # Set model to eval mode
    model.eval()

    # Iterate over data. Show a progress bar.
    #with tqdm(total=len(loader)) as pbar:
    for i, (inputs, labels) in enumerate(loader):

        # Prepare inputs and labels
        inputs = inputs.to(device)
        labels = labels.to(device)

        with torch.no_grad():

            outputs = model(inputs)

        # Check if binary or multi-class
        if outputs.shape[-1] == 1:
            preds = (outputs > 0).squeeze()
        else:
            preds = torch.argmax(outputs, dim=1)

        # Update accuracy counters
        n_correct += (preds == labels).float().sum()
        n_predicted += len(labels)

        # Update prediction lists
        for lab in labels.cpu().detach().numpy():
            total_labels.append(lab)
        for pred in preds.cpu().detach().numpy():
            total_predictions.append(pred)

    # Return labels, predictions, accuracy and loss
    return total_labels, total_predictions, (n_correct/n_predicted).cpu().detach().numpy()