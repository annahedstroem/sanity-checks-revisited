from typing import Dict, Optional, List
import numpy as np
import torch
import torchvision
from zennit import canonizers, composites, rules, attribution
from zennit import torchvision as zvision
from zennit import types as ztypes
from quantus import AVAILABLE_XAI_METHODS_CAPTUM

from .utils import create_masks_for_dataset


def get_zennit_canonizer(model):
    """
    Checks the type of model and selects the corresponding zennit canonizer
    """

    # ResNet
    if isinstance(model, torchvision.models.ResNet):
        return zvision.ResNetCanonizer

    # VGG
    if isinstance(model, torchvision.models.VGG):
        return zvision.VGGCanonizer

    # default fallback (only the above types have specific canonizers in zennit for now)
    return canonizers.SequentialMergeBatchNorm


class Epsilon(composites.LayerMapComposite):
    """An explicit composite using the epsilon rule for all layers

    Parameters
    ----------
    epsilon: callable or float, optional
        Stabilization parameter for the ``Epsilon`` rule. If ``epsilon`` is a float, it will be added to the
        denominator with the same sign as each respective entry. If it is callable, a function ``(input: torch.Tensor)
        -> torch.Tensor`` is expected, of which the output corresponds to the stabilized denominator. Note that this is
        called ``stabilizer`` for all other rules.
    stabilizer: callable or float, optional
        Stabilization parameter for rules other than ``Epsilon``. If ``stabilizer`` is a float, it will be added to the
        denominator with the same sign as each respective entry. If it is callable, a function ``(input: torch.Tensor)
        -> torch.Tensor`` is expected, of which the output corresponds to the stabilized denominator.
    layer_map: list[tuple[tuple[torch.nn.Module, ...], Hook]]
        A mapping as a list of tuples, with a tuple of applicable module types and a Hook. This will be prepended to
        the ``layer_map`` defined by the composite.
    zero_params: list[str], optional
        A list of parameter names that shall set to zero. If `None` (default), no parameters are set to zero.
    canonizers: list[:py:class:`zennit.canonizers.Canonizer`], optional
        List of canonizer instances to be applied before applying hooks.
    """

    def __init__(
        self,
        epsilon=1e-6,
        stabilizer=1e-6,
        layer_map=None,
        zero_params=None,
        canonizers=None,
    ):
        if layer_map is None:
            layer_map = []

        rule_kwargs = {"zero_params": zero_params}
        layer_map = (
            layer_map
            + composites.layer_map_base(stabilizer)
            + [
                (ztypes.Convolution, rules.Epsilon(epsilon=epsilon, **rule_kwargs)),
                (torch.nn.Linear, rules.Epsilon(epsilon=epsilon, **rule_kwargs)),
            ]
        )
        super().__init__(layer_map=layer_map, canonizers=canonizers)


class ZPlus(composites.LayerMapComposite):
    """
    An explicit composite using the epsilon rule for all layers

    Parameters
    ----------
    stabilizer: callable or float, optional
        Stabilization parameter for rules other than ``Epsilon``. If ``stabilizer`` is a float, it will be added to the
        denominator with the same sign as each respective entry. If it is callable, a function ``(input: torch.Tensor)
        -> torch.Tensor`` is expected, of which the output corresponds to the stabilized denominator.
    layer_map: list[tuple[tuple[torch.nn.Module, ...], Hook]]
        A mapping as a list of tuples, with a tuple of applicable module types and a Hook. This will be prepended to
        the ``layer_map`` defined by the composite.
    zero_params: list[str], optional
        A list of parameter names that shall set to zero. If `None` (default), no parameters are set to zero.
    canonizers: list[:py:class:`zennit.canonizers.Canonizer`], optional
        List of canonizer instances to be applied before applying hooks.
    """

    def __init__(
        self, stabilizer=1e-6, layer_map=None, zero_params=None, canonizers=None
    ):
        if layer_map is None:
            layer_map = []

        rule_kwargs = {"zero_params": zero_params}
        layer_map = (
            layer_map
            + composites.layer_map_base(stabilizer)
            + [
                (
                    ztypes.Convolution,
                    rules.ZPlus(stabilizer=stabilizer, **rule_kwargs),
                ),
                (torch.nn.Linear, rules.ZPlus(stabilizer=stabilizer, **rule_kwargs)),
            ]
        )
        super().__init__(layer_map=layer_map, canonizers=canonizers)


def setup_xai_methods_captum(
    xai_methods: List[str],
    x_batch: np.array,
    gc_layer: Optional[str] = None,
    img_size: int = 28,
    nr_channels: int = 1,
    nr_segments: int = 25,
) -> Dict:

    captum_methods = {
        "Gradient": {},
        "Saliency": {},
        "DeepLift": {},
        "GradientShape": {},
        "InputXGradient": {},
        "LayerGradCam": {
            "gc_layer": gc_layer,
            "interpolate": (img_size, img_size),
            "interpolate_mode": "bilinear",
            "xai_lib": "captum",
        },
        "Occlusion": {
            "window": (nr_channels, int(img_size / 4), int(img_size / 4)),
            "xai_lib": "captum",
        },
    }

    captum_methods["KernelShap"] = {
        "n_samples": 10,
        "xai_lib": "captum",
        "return_input_shape": True,
        "perturbations_per_eval": 1,  # len(x_batch)//10,
        "feature_mask": create_masks_for_dataset(
            torch.randn((len(x_batch), nr_channels, img_size, img_size)),
            nr_segments=nr_segments,
        ),
    }

    return {
        xai_method: captum_methods.get(xai_method, {"xai_lib": "captum"})
        for xai_method in xai_methods
        if xai_method in AVAILABLE_XAI_METHODS_CAPTUM
    }


def setup_xai_methods_zennit(
    xai_methods: List[str], model: torch.nn.Module, device: Optional[str] = None
) -> Dict[str, dict]:

    zennit_methods = {
        "SmoothGrad": {
            "xai_lib": "zennit",
            "attributor": attribution.SmoothGrad,
            "attributor_kwargs": {"n_iter": 20, "noise_level": 0.1},
            "canonizer": None,
            "composite": None,
            "canonizer_kwargs": {},
            "composite_kwargs": {},
            "device": device,
        },
        "IntegratedGradients": {
            "xai_lib": "zennit",
            "attributor": attribution.IntegratedGradients,
            "attributor_kwargs": {
                "n_iter": 20,
            },
            "canonizer": None,
            "composite": None,
            "canonizer_kwargs": {},
            "composite_kwargs": {},
            "device": device,
        },
        "LRP-Eps": {
            "xai_lib": "zennit",
            "attributor": attribution.Gradient,
            "composite": Epsilon,
            "canonizer": get_zennit_canonizer(model),
            "canonizer_kwargs": {},
            "composite_kwargs": {"stabilizer": 1e-6, "epsilon": 1e-6},
            "device": device,
        },
        "LRP-Z+": {
            "xai_lib": "zennit",
            "attributor": attribution.Gradient,
            "composite": ZPlus,
            "canonizer": get_zennit_canonizer(model),
            "canonizer_kwargs": {},
            "composite_kwargs": {
                "stabilizer": 1e-6,
            },
            "device": device,
        },
        "Guided-Backprop": {
            "xai_lib": "zennit",
            "attributor": attribution.Gradient,
            "canonizer": None,
            "composite": composites.GuidedBackprop,
            "canonizer_kwargs": {},
            "composite_kwargs": {},
            "device": device,
        },
        "Gradient": {
            "xai_lib": "zennit",
            "attributor": attribution.Gradient,
            "canonizer": None,
            "composite": None,
            "canonizer_kwargs": {},
            "composite_kwargs": {},
            "device": device,
        },
        "Saliency": {
            "xai_lib": "zennit",
            "attributor": attribution.Gradient,
            "canonizer": None,
            "composite": None,
            "canonizer_kwargs": {},
            "composite_kwargs": {},
            "device": device,
        },
    }

    return {
        xai_method: zennit_methods[xai_method]
        for xai_method in xai_methods
        if xai_method in zennit_methods
    }
