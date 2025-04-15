# eIP: Evidential Deep Learning for Interatomic Potentials

**Official implementation for the paper: "Evidential Deep Learning for Interatomic Potentials ".**

* [Overview](#overview)
* [System Requirements](#system-requirements)
* [Installation Guide](#installation-guide)
* [How to run this code](#how-to-run-this-code)


# Overview

Machine learning interatomic potentials (MLIPs) have been widely used to facilitate large-scale molecular simulations with accuracy comparable to ab initio methods. In practice, MLIP-based molecular simulations often encounter the issue of collapse due to reduced prediction accuracy for out-of-distribution (OOD) data. Addressing this issue requires enriching the training dataset through active learning, where uncertainty serves as a critical indicator for identifying and collecting OOD data. However, existing uncertainty quantification (UQ) methods tend to involve either expensive computations or compromise prediction accuracy. In this work, we introduce evidential deep learning for interatomic potentials (eIP) with a physics-inspired design. Our experiments indicate that eIP provides reliable UQ results without significant computational overhead or decreased prediction accuracy, consistently outperforming other UQ methods across a  variety of datasets. Furthermore, we demonstrate the applications of eIP in exploring diverse atomic configurations, using examples including water and universal potentials. These results highlight the potential of eIP as a robust and efficient alternative for UQ in molecular simulations.

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
 pip install torch torchvision torchaudio
 pip install torch_geometric
 pip install pyg_lib torch_scatter torch_sparse torch_cluster torch_spline_conv -f https://data.pyg.org/whl/torch-2.4.0+cu121.html
 pip install ase
 pip install -U scikit-learn
```


# How to run this code:


## Usage

1. Clone the repository
2. Install the required dependencies
3. Run the notebooks for demonstration:
   - `eIP_silica.ipynb` for uncertainty prediction
   - `md_udd.ipynb` for molecular dynamics simulation


## Repository Structure

```
├── calculator.py # Energy and force calculator 
├── checkpoint/ # Directory containing model weight
├── dataset/ # Directory containing datasets for training and testing
│   ├── silica_test.pt
│   └── silica_train.pt
├── dig/ # Directory containing utilities scripts
├── eIP_silica.ipynb # A Jupyter Notebook for uncertainty calculation demonstrations
├── LiFePO4.cif # Crystal information file for LiFePO4
├── PaiNN_md.py # Implementation of the PaiNN model using in MD
├── PaiNN.py # Implementation of the PaiNN model
├── PDMS.cif # Crystal information file for PDMS
├── run.py # Main execution script
├── test_eIP_silica.py # Uncertainty evaluation script for the silica glass dataset
├── md_udd.ipynb # 用于计算不确定度计算的展示notebook
├── test_md.py # Molecular dynamics testing script
├── train_eip.py # Model training script
└── udd_run.py # Script for uncertainty-driven dynamics
```


### Uncertainty Prediction

This part demonstrates the uncertainty prediction process for silica-glass datasets. To run the uncertainty prediction:

```bash
jupyter notebook eIP_silica.ipynb
```

This notebook demonstrates the complete uncertainty prediction workflow for silica-glass materials, including eIP training, result testing, and validation.

### Molecular Dynamics Simulation

This section performs molecular dynamics simulations for both materials using a universal potential function model, including conventional MD and uncertainty-driven dynamics (UDD).

1. PDMS (Polydimethylsiloxane)
2. LiFePO4 (Lithium Iron Phosphate)

To run the molecular dynamics simulation:

```bash
jupyter notebook md_udd.ipynb
```

This notebook demonstrates the molecular dynamics simulation process for both materials.




## Contact

For any questions or issues, please open an issue in the repository.
