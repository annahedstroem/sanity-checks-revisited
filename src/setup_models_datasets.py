"""This module contains functions for setting up experiments."""

from typing import Dict
import numpy as np
import torch
import torchvision
from torchvision.models.resnet import ResNet18_Weights
from torchvision.models.vgg import VGG16_Weights

from metaquantus import LeNet


def setup_experiments(
    dataset_name: str,
    path_assets: str,
    device: torch.device,
    suffix: str = "_random",
) -> Dict[str, dict]:
    """
    Setup dataset-specific models and data for MPRT, SmoothMPRT and EfficientMPRT evaluation.

    Parameters
    ----------
    dataset_name : str
        The name of the dataset ('MNIST', 'fMNIST', 'ImageNet').

    path_assets : str
        The path to the assets directory containing models and test sets.

    device : torch.device
        The device to load the models onto (e.g., 'cpu' or 'cuda').

    suffix : str, optional
        Suffix for dataset-specific assets.

    Returns
    -------
    dict
        A dictionary containing settings for eMPRT evaluation.
    """

    SETTINGS = {}

    if dataset_name == "MNIST":
        # Paths.
        path_mnist_model = path_assets + f"models/mnist_lenet"
        path_mnist_assets = path_assets + f"test_sets/mnist_test_set.npy"

        # Load model.
        model_mnist = LeNet()
        model_mnist.load_state_dict(torch.load(path_mnist_model, map_location=device))

        # Load data.
        assets_mnist = np.load(path_mnist_assets, allow_pickle=True).item()
        x_batch_mnist = assets_mnist["x_batch"]
        y_batch_mnist = assets_mnist["y_batch"]
        s_batch_mnist = assets_mnist["s_batch"]

        # Add to settings.
        SETTINGS["MNIST"] = {
            "x_batch": x_batch_mnist,
            "y_batch": y_batch_mnist,
            "s_batch": s_batch_mnist,
            "models": {"LeNet": model_mnist},
            "gc_layers": {"LeNet": "list(model.named_modules())[3][1]"},
            "estimator_kwargs": {
                "features": 28 * 2,
                "num_classes": 10,
                "img_size": 28,
                "percentage": 0.1,
                "nr_channels": 1,
                "patch_size": 28 * 2,
                "perturb_baseline": "uniform",
                "std_adversary": 2.0,
            },
        }

    elif dataset_name == "fMNIST":
        # Paths.
        path_fmnist_model = path_assets + f"models/fmnist_lenet"
        path_fmnist_assets = path_assets + f"test_sets/fmnist_test_set.npy"

        # Load model.
        model_fmnist = LeNet()
        model_fmnist.load_state_dict(torch.load(path_fmnist_model, map_location=device))

        # Load data.
        assets_fmnist = np.load(path_fmnist_assets, allow_pickle=True).item()
        x_batch_fmnist = assets_fmnist["x_batch"]
        y_batch_fmnist = assets_fmnist["y_batch"]
        s_batch_fmnist = assets_fmnist["s_batch"]

        # Add to settings.
        SETTINGS["fMNIST"] = {
            "x_batch": x_batch_fmnist,
            "y_batch": y_batch_fmnist,
            "s_batch": s_batch_fmnist,
            "models": {"LeNet": model_fmnist},
            "gc_layers": {"LeNet": "list(model.named_modules())[3][1]"},
            "estimator_kwargs": {
                "features": 28 * 2,
                "num_classes": 10,
                "img_size": 28,
                "percentage": 0.1,
                "nr_channels": 1,
                "patch_size": 28 * 2,
                "perturb_baseline": "uniform",
                "std_adversary": 2.0,
            },
        }

    elif dataset_name == "ImageNet":
        # Paths.
        path_imagenet_assets = path_assets + f"test_sets/imagenet_test_set{suffix}.npy"

        # Load data.
        assets_imagenet = np.load(path_imagenet_assets, allow_pickle=True).item()
        x_batch = assets_imagenet["x_batch"]
        y_batch = assets_imagenet["y_batch"]
        s_batch = assets_imagenet["s_batch"]

        # Add to settings.
        SETTINGS["ImageNet"] = {
            "x_batch": x_batch,
            "y_batch": y_batch,
            "s_batch": s_batch,
            "models": {
                "ResNet18": torchvision.models.resnet18(
                    weights=ResNet18_Weights.DEFAULT
                ).eval(),
                "VGG16": torchvision.models.vgg16(weights=VGG16_Weights.DEFAULT).eval(),
            },
            "gc_layers": {
                "ResNet18": "list(model.named_modules())[61][1]",
                "VGG16": "model.features[-2]",
            },
            "estimator_kwargs": {
                "num_classes": 1000,
                "img_size": 224,
                "features": 224 * 4,
                "percentage": 0.1,
                "nr_channels": 3,
                "patch_size": 224 * 2,
                "perturb_baseline": "uniform",
                "std_adversary": 0.5,
            },
        }

    else:
        raise ValueError("Provide a supported dataset {'MNIST', 'fMNIST', 'ImageNet'}.")

    return SETTINGS
