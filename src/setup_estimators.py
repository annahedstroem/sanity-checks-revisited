from typing import Callable, Optional, Dict, Any, Callable
import numpy as np
from quantus import MPRT, SmoothMPRT, EfficientMPRT
from quantus import similarity_func, complexity_func, normalise_func


def setup_estimators(
    layer_order: str = "bottom_up",
    noise_magnitude: float = 0.1,
    nr_samples: Optional[int] = 50,
    complexity_func: Callable = complexity_func.discrete_entropy,
    similarity_func_emprt: Callable = similarity_func.correlation_spearman,
    similarity_func_mprt: Callable = similarity_func.ssim,
    n_bins: int = 100,
    abs: bool = False,
    normalise: bool = True,
    normalise_func: Callable = normalise_func.normalise_by_average_second_moment_estimate,
    return_aggregate: bool = False,
) -> Dict[str, Dict[str, Any]]:
    """
    Set up a dictionary of estimators for final scoring in the MetaQuantus framework.

    Parameters
    ----------
    layer_order : str, optional
        The order in which layers will be considered during scoring. Default is "bottom_up".

    noise_magnitude : float, optional
        The magnitude of noise to apply in SmoothMPRT. Default is 0.1.

    nr_samples : int, optional
        The number of samples to use in the estimation. Default is 50.

    complexity_func : Callable, optional
        The complexity function used for estimation. Default is `complexity_func.discrete_entropy`.

    similarity_func_emprt : Callable, optional
        The similarity function used in EfficientMPRT. Default is `similarity_func.correlation_spearman`.

    similarity_func_mprt : Callable, optional
        The similarity function used in MPRT and SmoothMPRT. Default is `similarity_func.ssim`.

    n_bins : int, optional
        The number of bins used in complexity estimation. Default is 100.

    abs : bool, optional
        Whether to use absolute values for scores. Default is False.

    normalise : bool, optional
        Whether to normalise scores. Default is True.

    normalise_func : Callable, optional
        The normalisation function to use. Default is `normalise_func.normalise_by_average_second_moment_estimate`.

    return_aggregate : bool, optional
        Whether to return an aggregate score. Default is False.

    Returns
    -------
    Dict[str, Dict[str, Any]]
        A dictionary containing different estimators for final scoring in the MetaQuantus framework.

    Notes
    -----
    - The returned dictionary has three estimators for "Randomisation": "EfficientMPRT", "SmoothMPRT", and "MPRT".
    - Each estimator is configured with specified parameters and settings for scoring.

    """
    return {
        "Randomisation": {
            "eMPRT": {
                "init": EfficientMPRT(
                    complexity_func=complexity_func,
                    layer_order=layer_order,
                    similarity_func=similarity_func_emprt,
                    complexity_func_kwargs={"n_bins": n_bins},
                    skip_layers=True,
                    abs=abs,
                    normalise=normalise,
                    normalise_func=normalise_func,
                    return_aggregate=return_aggregate,
                    aggregate_func=np.mean,
                    disable_warnings=True,
                ),
                "score_direction": "higher",
            },
            "sMPRT": {
                "init": SmoothMPRT(
                    noise_magnitude=noise_magnitude,
                    nr_samples=nr_samples,
                    layer_order=layer_order,
                    return_last_correlation=True,
                    similarity_func=similarity_func_mprt,
                    skip_layers=True,
                    abs=abs,
                    normalise=normalise,
                    normalise_func=normalise_func,
                    return_aggregate=return_aggregate,
                    aggregate_func=np.mean,
                    disable_warnings=True,
                ),
                "score_direction": "lower",
            },
            "MPRT": {
                "init": MPRT(
                    layer_order=layer_order,
                    return_last_correlation=True,
                    similarity_func=similarity_func_mprt,
                    skip_layers=True,
                    abs=abs,
                    normalise=normalise,
                    normalise_func=normalise_func,
                    return_aggregate=return_aggregate,
                    aggregate_func=np.mean,
                    disable_warnings=True,
                ),
                "score_direction": "lower",
            },
        }
    }
