"""This module contains functions for setting up experiments."""

from typing import Dict
import os
import json
import numpy as np
import torch
import torch.utils.data as tdata
import torchvision
from torchvision.models.resnet import ResNet18_Weights
from torchvision.models.vgg import VGG16_Weights
from PIL import Image

from metaquantus import LeNet

from .configs import TV_MODEL_MAP, DATALOADER_MAPPING, TRANSFORM_MAP

class CustomDataset(tdata.Dataset):

    def __init__(self, root, transform, **kwargs):
        """
        Dataset for downloaded Data.
        Expected folder layout is
            root/mode/readable-class-name/sample.*
        """

        # Stores the arguments for later use
        self.transform = transform

        self.mode = kwargs.get("mode", "train")
        self.classes = kwargs.get("classes", "all")
        self.accepted_filetypes = kwargs.get("accepted_filetypes", ["png", "jpeg", "jpg"])
        self.labelmap_path = kwargs.get("labelmap_path", None)

        assert isinstance(self.mode, str)
        assert self.labelmap_path is not None

        self.root = os.path.join(root, self.mode)

        # Loads the label map into memory
        with open(self.labelmap_path) as label_map_file:
            # label map should map from class name to class idx
            self.label_map = json.load(label_map_file)

        # create list of (sample, label) tuples.
        self.samples = []

        print("DATA_ROOT", self.root)

        # Build samples with correct classes first
        for cl in self.label_map.keys():
            cl_dir = os.path.join(self.root, cl)
            if os.path.exists(cl_dir) and self.classes == "all" or cl in self.classes:
                for fname in [f for f in os.listdir(cl_dir) if os.path.isfile(os.path.join(cl_dir, f))]:
                    if fname.split(".")[-1].lower() in self.accepted_filetypes:
                        self.samples.append((os.path.join(cl_dir, fname), self.label_map[cl]["label"]))


    def __len__(self):
        """
        Retrieves the number of samples in the dataset.

        Returns
        -------
            int
                Returns the number of samples in the dataset.
        """

        return len(self.samples)

    def __getitem__(self, index):
        """
        Retrieves the element with the specified index.

        Parameters
        ----------
            index: int
                The index of the element that is to be retrieved.

        Returns
        -------
            Sample
                Returns the sample with the specified index
        """

        # Gets the path name and the WordNet ID of the sample
        file, label = self.samples[index]

        # Loads the image from file
        sample = Image.open(file)
        sample = sample.convert('RGB')

        # If the user specified a transform for the samples, then it is applied
        sample = self.transform(sample)

        # Returns the sample
        return sample, label

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

def get_model(model_name, device):
    """
    Gets the correct model
    """

    # Check if model_name is supported
    if model_name not in TV_MODEL_MAP:
        raise ValueError("Model '{}' is not supported.".format(model_name))

    # Build model
    if model_name in TV_MODEL_MAP:
        model = TV_MODEL_MAP[model_name](pretrained=True)

    # Return model on correct device
    return model.to(device)

DATASET_MAPPING = {
    "imagenet": CustomDataset,
}

def get_dataloader(dataset_name, dataset, batch_size, shuffle):
    """
    selects the correct dataloader for the dataset
    """

    # Check if dataset_name is valid
    if dataset_name not in DATALOADER_MAPPING:
        raise ValueError("Dataloader for dataset '{}' not supported.".format(dataset_name))

    # Load correct dataloader
    dataloader = DATALOADER_MAPPING[dataset_name](
        dataset=dataset,
        batch_size=batch_size,
        shuffle=shuffle,
    )

    # Return dataset
    return dataloader

def get_dataset(dataset_name, root_path, transform, mode, **kwargs):
    """
    gets the specified dataset and saves it
    """

    # Check if mode is valid
    if mode not in ["train", "test"]:
        raise ValueError("Mode '{}' not supported. Mode needs to be one of 'train', 'test'".format(mode))

    # Map mode (kinda illegal but so that imagenet works)
    if (dataset_name == "imagenet" or dataset_name == "imagenet-bbox" or dataset_name == "pascalvoc") and mode == "test":
        mode = "val"

    # Check if dataset_name is valid
    if dataset_name not in DATASET_MAPPING:
        raise ValueError("Dataset '{}' not supported.".format(dataset_name))

    # Adapt root_path
    if DATASET_MAPPING[dataset_name] not in [CustomDataset]:
        root = os.path.join(root_path, dataset_name)
    else:
        root = root_path

    # Load correct dataset
    if dataset_name not in ["mnist", "fashionmnist", "cifar10", "cifar10-transfer", "cifar100"]:
        dataset = DATASET_MAPPING[dataset_name](
            root = root,
            transform = transform,
            **{
                **kwargs,
                **{
                    "download": True,
                    "train": mode == "train",
                    "mode": mode,
                },
            }
        )
    else:
        dataset = DATASET_MAPPING[dataset_name](
            root=root,
            transform=transform,
            download = True,
            train = mode == "train",
            **kwargs
        )

    # Return dataset
    return dataset

def get_transforms(dataset_name, mode):
    """
    Gets the correct transforms for the dataset
    """

    # Check if dataset_name is supported
    if dataset_name not in TRANSFORM_MAP:
        raise ValueError("Dataset '{}' not supported.".format(dataset_name))

    # Combine transforms
    transforms = torchvision.transforms.Compose(TRANSFORM_MAP[dataset_name][mode])

    # Return transforms
    return transforms