"""This module contains the configuration for plotting."""
import matplotlib.pyplot as plt
import matplotlib.font_manager as font_manager
import numpy as np
import torch.utils.data as tdata
import torchvision
import torchvision.transforms as T

# Set font properties.
font_path = plt.matplotlib.get_data_path() + "/fonts/ttf/cmr10.ttf"
cmfont = font_manager.FontProperties(fname=font_path)
plt.rcParams["font.family"] = "serif"
plt.rcParams["font.serif"] = cmfont.get_name()
plt.rcParams["mathtext.fontset"] = "cm"

# Set font size.
plt.rcParams["font.size"] = 15

# Disable unicode minus.
plt.rcParams["axes.unicode_minus"] = False

# Use mathtext for axes formatters.
plt.rcParams["axes.formatter.use_mathtext"] = True

# General plotting.
palette = plt.cm.get_cmap("tab20")(np.linspace(0, 1, 20))
std_alpha = 0.2

LAYER_ORDER_MAP = {"top_down": "Top-Down", "bottom_up": "Bottom-Up"}

COLOR_MAP = {
    "Model": "black",
    "Saliency": palette[1],
    "Gradient": palette[2],
    "LayerGradCam": palette[3],
    "SmoothGrad": palette[4],
    "IntegratedGradients": palette[5],
    "LRP-Eps": palette[6],
    "LRP-Z+": palette[7],
    "Guided-Backprop": palette[8],
    "GradientShap": palette[9],
    "InputXGradient": palette[10],
    "Control Var. Random Uniform": "darkblue",
}
LABEL_MAP = {
    "Model": "Model",
    "Saliency": "Saliency",
    "Gradient": "Gradient",
    "LayerGradCam": "GradCAM",
    "SmoothGrad": "SmoothGrad",
    "IntegratedGradients": "IntegratedGradients",
    "LRP-Eps": r"LRP-$\varepsilon$",
    "LRP-Z+": r"LRP-$z^+$",
    "Guided-Backprop": "Guided-Backprop",
    "GradientShap": "GradientSHAP",
    "InputXGradient": "InputXGradient",
    "Control Var. Random Uniform": "Random Attribution",
}

PLOTTING_COLOURS = {
    "eMPRT_ImageNet_VGG16": "#542788",
    "eMPRT_ImageNet_ResNet18": "#8073ac",
    "eMPRT_MNIST_LeNet": "#b2abd2",
    "eMPRT_fMNIST_LeNet": "#d8daeb",
    "sMPRT_ImageNet_VGG16": "#b35806",
    "sMPRT_ImageNet_ResNet18": "#e08214",
    "sMPRT_MNIST_LeNet": "#fdb863",
    "sMPRT_fMNIST_LeNet": "#fee0b6",
    "MPRT_ImageNet_VGG16": "#1b7837",
    "MPRT_ImageNet_ResNet18": "#5aae61",
    "MPRT_MNIST_LeNet": "#a6dba0",
    "MPRT_fMNIST_LeNet": "#d9f0d3",
}

HATCH_MAP = {
    "top_down": "///", 
    "bottom_up": ".",
}
LINESTYLE_ORDER = ["dashed", "solid"]

LINESTYLE_MAP = {
    "1": "dashed",
    "50": "solid",
}

TV_MODEL_MAP = {
    "resnet50": torchvision.models.resnet50,
    "resnet34": torchvision.models.resnet34,
    "resnet18": torchvision.models.resnet18,
    "densenet121": torchvision.models.densenet121,
    "vgg16": torchvision.models.vgg16,
}

DATALOADER_MAPPING = {
    "imagenet": tdata.DataLoader,
}

TRANSFORM_MAP = {
    "imagenet": {
            "train": [T.RandomResizedCrop(224), T.RandomHorizontalFlip(), T.ToTensor(), T.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))],
            "test": [T.Resize((224, 224)),  T.ToTensor(), T.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))],
            "val": [T.Resize((224, 224)),  T.ToTensor(), T.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))],
    },
}