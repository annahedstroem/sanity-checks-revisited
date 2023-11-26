"""This module contains the script for obtaining the results associated with the sanity checks revisited benchmarking experiment."""

import os
import warnings
import argparse
import torch
import sys
from typing import List, Callable

from metaquantus import MetaEvaluation, MetaEvaluationBenchmarking
from metaquantus import setup_test_suite

from .setup_models_datasets import setup_experiments
from .setup_explanations import setup_xai_methods_zennit, setup_xai_methods_captum


def run_benchmarking_script(
    dataset_name: str,
    model_name: str,
    K: str,
    iters: str,
    start_idx: str,
    end_idx: str,
    path_assets: str,
    path_results: str,
    folder: str,
    xai_methods: List[str],
    setup_metrics: Callable,
    normalise: bool = True,
    fname_addition: str = "",
):

    # Setting sys.argv with the provided command-line arguments.
    sys.argv = [
        "experiments/run_benchmarking.py",
        "--dataset_name",
        dataset_name,
        "--model_name",
        model_name,
        "--K",
        K,
        "--iters",
        iters,
        "--start_idx",
        start_idx,
        "--end_idx",
        end_idx,
        "--folder",
        folder,
        "--PATH_ASSETS",
        path_assets,
        "--PATH_RESULTS",
        path_results,
    ]

    ######################
    # Parsing arguments. #
    ######################

    print(f"Running from path: {os.getcwd()}")
    warnings.filterwarnings("ignore", category=FutureWarning)
    warnings.filterwarnings("ignore", category=UserWarning)

    def parse_arguments():
        parser = argparse.ArgumentParser()
        parser.add_argument("--dataset_name")  # , default="MNIST")
        parser.add_argument("--model_name")  # , default="LeNet")
        parser.add_argument("--K")  # , default=5)
        parser.add_argument("--iters")  # , default=3)
        parser.add_argument("--start_idx")
        parser.add_argument("--end_idx")
        parser.add_argument("--folder")  # , default="benchmarking/")
        parser.add_argument(
            "--PATH_ASSETS"
        )  # , default="/content/drive/MyDrive/Projects/MetaQuantus/assets/")
        parser.add_argument(
            "--PATH_RESULTS"
        )  # , default="content/drive/MyDrive/Projects/neurips-xai-workshop/"
        args = parser.parse_args()

        return (
            str(args.dataset_name),
            str(args.model_name),
            int(args.K),
            int(args.iters),
            int(args.start_idx),
            int(args.end_idx),
            str(args.folder),
            str(args.PATH_ASSETS),
            str(args.PATH_RESULTS),
            f"{args.model_name}-{args.start_idx}-{args.end_idx}{fname_addition}",
        )

    # Get arguments.
    (
        dataset_name,
        model_name,
        K,
        iters,
        start_idx,
        end_idx,
        folder,
        PATH_ASSETS,
        PATH_RESULTS,
        fname,
    ) = parse_arguments()
    print(
        "Arguments:\n",
        dataset_name,
        model_name,
        K,
        iters,
        start_idx,
        end_idx,
        folder,
        PATH_ASSETS,
        PATH_RESULTS,
        fname,
    )

    #########
    # GPUs. #
    #########

    # Setting device on GPU if available, else CPU.
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("\nUsing device:", device)
    print("\t{torch.version.cuda}")

    # Additional info when using cuda.
    if device.type == "cuda":
        print(f"\t{torch.cuda.get_device_name(0)}")
        print("\tMemory Usage:")
        print(
            "\tAllocated:", round(torch.cuda.memory_allocated(0) / 1024 ** 3, 1), "GB"
        )
        print("\tCached:   ", round(torch.cuda.memory_cached(0) / 1024 ** 3, 1), "GB")

    ##############################
    # Dataset-specific settings. #
    ##############################

    # Get input, outputs settings.
    SETTINGS = setup_experiments(
        dataset_name=dataset_name, path_assets=PATH_ASSETS, device=device
    )
    dataset_settings = {dataset_name: SETTINGS[dataset_name]}
    estimator_kwargs = dataset_settings[dataset_name]["estimator_kwargs"]

    # Get analyser suite.
    analyser_suite = setup_test_suite(dataset_name=dataset_name)

    # Get model_name.
    model = dataset_settings[dataset_name]["models"][model_name].eval()

    # Drop other models.
    all_models = list(dataset_settings[dataset_name]["models"].keys())
    if len(all_models) > 1:
        other_models = [k for k in all_models if k != model_name]
        for k in other_models:
            del dataset_settings[dataset_name]["models"][k]
            del dataset_settings[dataset_name]["gc_layers"][k]

    # Reduce the number of samples.
    dataset_settings[dataset_name]["x_batch"] = dataset_settings[dataset_name][
        "x_batch"
    ][start_idx:end_idx]
    dataset_settings[dataset_name]["y_batch"] = dataset_settings[dataset_name][
        "y_batch"
    ][start_idx:end_idx]
    dataset_settings[dataset_name]["s_batch"] = dataset_settings[dataset_name][
        "s_batch"
    ][start_idx:end_idx]

    # Update model-specific xai parameters.
    xai_methods_with_kwargs = {
        **setup_xai_methods_captum(
            xai_methods=xai_methods,
            x_batch=dataset_settings[dataset_name]["x_batch"][start_idx:end_idx],
            gc_layer=dataset_settings[dataset_name]["gc_layers"][model_name],
            img_size=estimator_kwargs["img_size"],
            nr_channels=estimator_kwargs["nr_channels"],
            nr_segments=50,
        ),
        **setup_xai_methods_zennit(xai_methods=xai_methods, model=model),
    }

    print(model_name, xai_methods_with_kwargs)

    # Load metrics.
    estimators = setup_metrics()

    ###########################
    # Benchmarking settings. #
    ###########################

    # Define master!
    master = MetaEvaluation(
        test_suite=analyser_suite,
        xai_methods=xai_methods_with_kwargs,
        iterations=iters,
        fname=fname,
        nr_perturbations=K,
    )

    # Benchmark!
    benchmark = MetaEvaluationBenchmarking(
        master=master,
        estimators=estimators,
        experimental_settings=dataset_settings,
        path=PATH_RESULTS,
        folder=folder,
        write_to_file=True,
        keep_results=True,
        channel_first=True,
        softmax=True,
        save=True,
        device=device,
    )()

    return benchmark
