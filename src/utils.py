from typing import Optional, Tuple, Dict, Any

import numpy as np
import matplotlib.pyplot as plt
import numpy as np
import torch
import skimage
from .configs import COLOR_MAP, LABEL_MAP


def get_indices(batch_size: int = 25, dataset_length: int = 300):
    """
    Generate a list of index ranges for splitting a dataset into batches.

    Parameters:
    - batch_size (int, optional): The size of each batch. Default is 25.
    - dataset_length (int, optional): The total length of the dataset. Default is 300.

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
                label=LABEL_MAP[method] + " (SSIM)",
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
