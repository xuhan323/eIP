# eIP


**This is the official implementation for the paper: "Evidential Deep Learning for Interatomic Potential ".**

* [Overview](#overview)
* [System Requirements](#system-requirements)
* [Installation Guide](#installation-guide)
* [How to run this code](#how-to-run-this-code)

# Overview

Machine Learning Interatomic Potentials (MLIPs) are models that utilize machine learning techniques to fit interatomic potential functions, with training data derived from ab initio methods. Consequently, MLIPs can achieve ab initio potential function accuracy with significantly faster inference times and reduced computational resource consumption. However, the datasets required for training MLIPs at ab initio accuracy are inherently resource-intensive and cannot encompass all possible configurations. When MLIP models trained on these datasets are employed in molecular dynamics (MD) simulations, they may encounter out-of-distribution (OOD) data, leading to a collapse of the MD simulation. To mitigate this issue, active learning approaches can be employed, iteratively sampling OOD data to enrich the training database. Nonetheless, conventional methods often require substantial time or result in decreased MLIP model accuracy. We propose a novel uncertainty output method that effectively balances speed and accuracy, demonstrating excellent performance.

# System Requirements

## Hardware requirements

A GPU is required for running this code base, RTX 3090 card and RTX 4090 have been tested.

## Software requirements

### OS Requirements

This code base is supported for Linux and has been tested on the following systems:

* **Linux: Ubuntu 20.04**

### Python Version

Python 3.9.15 has been tested.

# Installation Guide:

### Install dependencies

```
 conda install mamba -n base -c conda-forge
 mamba env create -f environment.yaml
 conda activate eIP


```

# How to run this code:

### Notebook (Demo)

In `eIP.ipynb`,  we have demonstrated the training process of the eIP model using the small molecule dataset as an example, as described in the article. The modifications to the parameters and the selection of datasets involved in the article are as follows:

By modifying the test.py, you can achieve changes to the hyperparameters in eIP.

### Download the pre-train dataset to Pretrain_dataset and save processed files to Pretrain_dataset:

```
 python xyz2pt.py
```

### Pre-training on MD17

**The model is pre-trained on the MD17 dataset, which consists of MD trajectories of small organic molecules with reference values of energy and forces calculated by ab initio molecular dynamics (AIMD) simulations. We use a pre-training dataset of 160,000 configurations sampled from CMD trajectories of eight different organic molecules.**

```
 python pretraining.py
```

### Fine-tuning on MD17 Datasets(Benzene2017 Dataset)

```
 python finetune.py
```

**For other datasets in MD17 Datasets, you can change the **`data_name` in the 15 line of the `finetune.py` (replace the 'benzene2017' with 'aspirin', 'ethanol', 'malonaldehyde', 'naphthalene', 'salicylic', 'toluene', 'uracil').
