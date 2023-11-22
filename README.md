This repository contains the code and experimental results for the paper **[Sanity Checks Revisited: An Exploration to Repair the Model Parameter Randomisation Test](https://openreview.net/forum?id=vVpefYmnsG)** by Hedström et al., 2023.

<!--[![Getting started!](https://colab.research.google.com/assets/colab-badge.svg)](anonymous)-->
<!--![Python version](https://img.shields.io/badge/python-3.7%20%7C%203.8%20%7C%203.9%20%7C%203.10%20%7C%203.11-blue.svg)-->
<!--[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)-->
<!--[![PyPI version](https://badge.fury.io/py/metaquantus.svg)](https://badge.fury.io/py/metaquantus)-->
<!--[![Python package](https://github.com/annahedstroem/MetaQuantus/actions/workflows/python-publish.yml/badge.svg)](https://github.com/annahedstroem/MetaQuantus/actions/workflows/python-publish.yml/badge.svg)-->
<!--[![Launch Tutorials](https://mybinder.org/badge_logo.svg)](anonymous)-->

## Citation

If you find this work or its companion paper interesting or useful in your research, use the following Bibtex annotation to cite us:

```bibtex
@inproceedings{hedstroem2023sanity,
    title={Sanity Checks Revisited: An Exploration to Repair the Model Parameter Randomisation Test},
    author={Anna Hedstr{\"o}m and Leander Weber and Sebastian Lapuschkin and Marina H{\"o}hne},
    booktitle={XAI in Action: Past, Present, and Future Applications},
    year={2023},
    url={https://openreview.net/forum?id=vVpefYmnsG}
}
```
<!--![Schematic visualisation of MPRT](https://raw.githubusercontent.com/annahedstroem/sanity-checks-revisited/main/motivation-3.png)-->


## Overview

The Model Parameter Randomisation Test (MPRT) is widely acknowledged in the eXplainable Artificial Intelligence (XAI) community for its well-motivated evaluative principle: that the explanation function should be sensitive to changes in the parameters of the model function. However, recent works have identified several methodological caveats for the empirical interpretation of MPRT. In this work, we introduce two adaptations to the original MPRT---Smooth MPRT and Efficient MPRT, where the former minimises the impact that noise has on the evaluation results and the latter circumvents the need for biased similarity measurements by re-interpreting the test through the explanation's rise in complexity, post-model randomisation. Our experimental results demonstrate improved metric reliability, for more trustworthy applications of XAI methods.

</p>
<p align="center">
  <img width="600" src="https://raw.githubusercontent.com/annahedstroem/sanity-checks-revisited/main/motivation-3.png">
</p>

Schematic visualisation of identified shortcomings (\emph{top}) and proposed solutions (\emph{bottom}) of the MPRT \cite{adebayo2018}. Solid arrows signify shortcomings directly addressed by our proposed metrics, while dashed arrows denote those addressed through ideas from previous work \cite{sundararajan2018,bindershort}. (a) The original MPRT evaluates an explanation method by randomising $f$'s parameters in a top-down, layer-by-layer manner and thereafter calculating explanation similarity $\rho(\ve, \hat{\ve})$ at each layer through comparing explanations $\ve$ of the original model $f$ and $\hat{\ve}$ of the randomised model $\hat{f}$. (b) \textit{Pre-processing}: normalisation and taking absolute attribution values significantly impact MPRT results, potentially deleting pertinent information about feature importance carried in the sign. (c) \textit{Layer-order}: top-down randomisation of layers in MPRT does not yield a fully random output, preserving properties of the unrandomised lower layers and thus affecting the evaluation of faithful explanations. (d) \textit{Similarity measures}: the pairwise similarity measures used in the original MPRT \cite{adebayo2018} are noise-sensitive, e.g., from gradient shattering and thus likely to impact evaluation rankings of XAI methods. (e) sMPRT extends MPRT by incorporating a preprocessing step that averages denoised attribution estimates over $N$ perturbed inputs, aiming to reduce noise in local explanation methods. (f) eMPRT reinterprets MPRT by evaluating the faithfulness of the attribution method by comparing its rise in complexity of a non- and fully random model. 



### Repo Instructions

- **Metrics Implementation**: Implemented in [Quantus](https://github.com/understandable-machine-intelligence-lab/Quantus), a popular toolkit for XAI evaluation.
- **Benchmarking**: Conducted using [MetaQuantus](https://github.com/annahedstroem/MetaQuantus/), a framework for meta-evaluating XAI metrics.
- **Source Code**: Contains Python functions added in `src`.
- **Notebooks**: Located in `nbs`, these include data generation for paper plots and benchmarking experiments.
- **Dependencies**: Listed in `requirements.txt`.

Note that these experiments require that [PyTorch](https://pytorch.org/) is installed on your machine.

## Package Requirements 

Required packages are:

```setup
python>=3.10.1
pytorch>=1.10.1
quantus>=0.3.2
metaquantus>=0.0.5
captum>=0.4.1
```

## Installation

Install the necessary packages using the provided [requirements.txt](https://annahedstroem/sanity-checks-revisited/blob/main/requirements.txt):

```bash
pip install -r requirements.txt
```
