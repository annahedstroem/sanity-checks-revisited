<br/><br/>
<p align="center">
  <img width="450" src="https://github.com/annahedstroem/sanity-checks-revisited/blob/394f166226e4ac415c6534e0e0441d8b3c9258f2/emprt_smprt_logo.png">
<!--<h3 align="center"><b>Evaluate the Explanation Faithfulness</b></h3>
<p align="center">
  PyTorch-->

  </p>

This repository contains the code and experiments for the paper **[Sanity Checks Revisited: An Exploration to Repair the Model Parameter Randomisation Test](https://openreview.net/forum?id=vVpefYmnsG)** by Hedström et al., 2023. 


<!--[![Getting started!](https://colab.research.google.com/assets/colab-badge.svg)](anonymous)-->
<!--![Python version](https://img.shields.io/badge/python-3.7%20%7C%203.8%20%7C%203.9%20%7C%203.10%20%7C%203.11-blue.svg)-->
<!--[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)-->
<!--[![PyPI version](https://badge.fury.io/py/metaquantus.svg)](https://badge.fury.io/py/metaquantus)-->
<!--[![Python package](https://github.com/annahedstroem/MetaQuantus/actions/workflows/python-publish.yml/badge.svg)](https://github.com/annahedstroem/MetaQuantus/actions/workflows/python-publish.yml/badge.svg)-->
<!--[![Launch Tutorials](https://mybinder.org/badge_logo.svg)](anonymous)-->

## Citation

If you find this work interesting or useful in your research, use the following Bibtex annotation to cite us:

```bibtex
@inproceedings{hedstroem2023sanity,
    title={Sanity Checks Revisited: An Exploration to Repair the Model Parameter Randomisation Test},
    author={Anna Hedstr{\"o}m and Leander Weber and Sebastian Lapuschkin and Marina H{\"o}hne},
    booktitle={XAI in Action: Past, Present, and Future Applications},
    year={2023},
    url={https://openreview.net/forum?id=vVpefYmnsG}
}
```
This work has been published in _[XAI in Action: Past, Present, and Future Applications](https://xai-in-action.github.io/)_ workshop at the 37th Conference on Neural Information Processing Systems (NeurIPS).

## Overview

The Model Parameter Randomisation Test (MPRT) is widely acknowledged in the eXplainable Artificial Intelligence (XAI) community for its well-motivated evaluative principle: that the explanation function should be sensitive to changes in the parameters of the model function. Recent studies, however, have pointed out practical limitations in MPRT's empirical application. To address these, we've introduced two adaptations: Smooth MPRT (sMPRT) and Efficient MPRT (eMPRT). sMPRT reduces noise effects in evaluations, while eMPRT avoids biased similarity measures by focusing on the complexity increase in explanations after model randomisation.

All evaluation metrics used in these experiments are implemented in [Quantus](https://github.com/understandable-machine-intelligence-lab/Quantus), a widely-used toolkit for XAI evaluation. Benchmarking is performed with tools from [MetaQuantus](https://github.com/annahedstroem/MetaQuantus/), a specialised framework for meta-evaluating XAI metrics.

The repository is organised for ease of use:
- The `src/` folder contains all necessary functions.
- The `nbs/` folder includes notebooks for generating the plots in the paper and for benchmarking experiments.

</p>
<p align="center">
  <img width="800" src="https://github.com/annahedstroem/sanity-checks-revisited/blob/33174dceeee19ef4bcfee5499b1436693c3121ea/motivation_3.png"> 
</p>
_Schematic visualisation of the original Model Parameter Randomisation Test (MPRT) (top), identified shortcomings (middle) and proposed solutions (bottom) of the MPRT [Adebayo et al., 2020](https://arxiv.org/pdf/1810.03292.pdf). Solid arrows signify shortcomings directly addressed by our proposed metrics, while dashed arrows denote those addressed through ideas from previous work [Sundararajan et al., 2018](https://arxiv.org/pdf/1806.04205.pdf), [Binder et al., 2022](https://arxiv.org/pdf/2211.12486.pdf). (a) The original MPRT evaluates an explanation method by randomising $f$'s parameters in a top-down, layer-by-layer manner and thereafter calculating explanation similarity $\rho(e, \hat{e})$ at each layer through comparing explanations $e$ of the original model $f$ and $\hat{e}$ of the randomised model $\hat{f}$. (b) Pre-processing: normalisation and taking absolute attribution values significantly impact MPRT results, potentially deleting pertinent information about feature importance carried in the sign. (c) Layer-order: top-down randomisation of layers in MPRT does not yield a fully random output, preserving properties of the unrandomised lower layers and thus affecting the evaluation of faithful explanations. (d) Similarity measures: the pairwise similarity measures used in the original MPRT [Adebayo et al., 2020](https://arxiv.org/pdf/1810.03292.pdf) are noise-sensitive, e.g., from gradient shattering and thus likely to impact evaluation rankings of XAI methods. (e) sMPRT extends MPRT by incorporating a preprocessing step that averages denoised attribution estimates over $N$ perturbed inputs, aiming to reduce noise in local explanation methods. (f) eMPRT reinterprets MPRT by evaluating the faithfulness of the attribution method by comparing its rise in complexity of a non- and fully random model._

## Installation

Install the necessary packages using the provided [requirements.txt](https://annahedstroem/sanity-checks-revisited/blob/main/requirements.txt):

```bash
pip install -r requirements.txt
```

## Package Requirements 

Required packages are:

```setup
python>=3.10.1
torch>=2.0.0
quantus>=0.5.0
metaquantus>=0.0.5
captum>=0.6.0
```

### Thank you

We hope our repository is beneficial to your work and research. If you have any feedback, questions, or ideas, please feel free to raise an issue in this repository. Alternatively, you can reach out to us directly via email for more in-depth discussions or suggestions. 

📧 Contact us: 
- Anna Hedström: [hedstroem.anna@gmail.com](mailto:hedstroem.anna@gmail.com)
- Leander Weber: [leander.weber@hhi.fraunhofer.de](mailto:leander.weber@hhi.fraunhofer.de)

Thank you for your interest and support!


