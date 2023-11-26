"""This module contains the configuration for plotting."""
import matplotlib.pyplot as plt
import matplotlib.font_manager as font_manager
import numpy as np

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

LAYER_ORDER_MAP = {"top_down": "top-down", "bottom_up": "bottom_up"}

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
