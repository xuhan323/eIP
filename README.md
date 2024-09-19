# eIP

**This is the official implementation for the paper: "Evidential Deep Learning for Interatomic Potential ".**

* [Overview](#overview)
* [System Requirements](#system-requirements)
* [Installation Guide](#installation-guide)
* [How to run this code](#how-to-run-this-code)

# Overview

Machine Learning Interatomic Potentials (MLIPs) are models that utilize machine learning techniques to fit interatomic potential functions, with training data derived from ab initio methods. Consequently, MLIPs can achieve ab initio potential function accuracy with significantly faster inference times and reduced computational resource consumption. However, the datasets required for training MLIPs at ab initio accuracy are inherently resource-intensive and cannot encompass all possible configurations. When MLIP models trained on these datasets are employed in molecular dynamics (MD) simulations, they may encounter out-of-distribution (OOD) data, leading to a collapse of the MD simulation. To mitigate this issue, active learning approaches can be employed, iteratively sampling OOD data to enrich the training database. Nonetheless, conventional methods often require substantial time or result in decreased MLIP model accuracy. We propose a novel uncertainty output method that effectively balances speed and accuracy, demonstrating excellent performance.

The UDD method enhances traditional MD simulations by leveraging uncertainty estimates from machine learning models to dynamically adjust simulation parameters. This implementation utilizes a neural network potential (PaiNN) to compute atomic forces and uncertainties during the simulation.

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

Alternatively, if limited by account permissions, network access, server system version, and other issues, you can try the following simpler installation methods through `pip`.

```
 pip3 install torch torchvision torchaudio
 pip install torch_geometric
 pip install pyg_lib torch_scatter torch_sparse torch_cluster torch_spline_conv -f https://data.pyg.org/whl/torch-2.4.0+cu121.html
 
```

### Time Consuming

Under normal network conditions, it takes 5-10 minutes, depending on fluctuations in network speed, to be faster or slower.


# How to run this code:

### Notebook (Demo)

In `eIP.ipynb`,  we have demonstrated the training process of the eIP model using the small molecule dataset as an example, as described in the article. The modifications to the parameters and the selection of datasets involved in the article are as follows:

By modifying the test.py, you can achieve changes to the hyperparameters in eIP.

### Time Consuming

In the demo file, each epoch takes approximately 40 seconds (tested on an RTX 4090), and this time may fluctuate depending on the complexity of the data (molecular configurations), the batch size, and other hyperparameters.

#### Parameters

```
device (torch.device): Device for computation.
train_dataset: Training data.
valid_dataset: Validation data.
test_dataset: Test data.
model: The eIP model.
loss_func (function): The used loss funtion for Machine Learning Interatomic Potential.
evaluation (function): The evaluation function. 
mol_name (str): The name of dataset or task
energy_trans (int, optinal): This value is used to adjust the zero point of potential energy, shifting the energy to around zero.
LAMBDA (int): The hyperparamers uese in eIP
THETA (int): The hyperparamers uese in eIP
q (int): The hyperparamers uese in eIP
epochs (int, optinal): Number of total training epochs. 
batch_size (int, optinal): Number of samples in each minibatch in training. (default: :obj:`32`)
vt_batch_size (int, optinal): Number of samples in each minibatch in validation/testing. (default: :obj:`32`)
lr (float, optinal): Initial learning rate. (default: :obj:`0.0005`)
lr_decay_factor (float, optinal): Learning rate decay factor. (default: :obj:`0.5`)
lr_decay_step_size (int, optinal): epochs at which lr_initial <- lr_initial * lr_decay_factor. (default: :obj:`50`)
energy_and_force (bool, optional): If set to :obj:`True`, will predict energy and take the minus derivative of the energy with respect to the atomic positions as predicted forces. (default: :obj:`False`)  
save_dir (str, optinal): The path to save trained models. If set to :obj:`''`, will not save the model. (default: :obj:`''`)
```

### UDD

All the code to run eIP-based UDD are in the folder 'udd-md'. To run a UDD MD simulation, use the `UDD_MD_run` function defined in `udd_run.py`. Modify the parameters as needed:

#### Python example:

```
from udd_run import UDD_MD_run
from ase.io import read

# Example usage
atoms = read('initial_structure.xyz')  # Load initial atomic configuration
Runtime = 1000  # Total runtime of the simulation in femtoseconds
PATH = '/path/to/model/weights.pth'  # Path to the trained model weights
Temp = 300  # Temperature of the system in Kelvin
sigma_cutoff = 0.1  # Cutoff value for uncertainty in atomic forces
filename = 'trajectory.xyz'  # Filename to save the trajectory
sampling = 20  # Interval for sampling and saving configurations
dt = 0.5  # Timestep for the simulation in femtoseconds
tau_t = 20  # Relaxation time for temperature coupling in femtoseconds

# Run the simulation
total_steps = UDD_MD_run(atoms, Runtime, PATH, Temp, sigma_cutoff, filename, sampling, dt, tau_t)
print(f"Simulation completed with {total_steps} steps.")
```

#### Parameters

```
atoms: ASE Atoms object containing the initial atomic configuration.
Runtime: Total runtime of the simulation in femtoseconds.
PATH: Path to the directory containing the trained model weights.
Temp: Temperature of the system in Kelvin.
sigma_cutoff: Cutoff value for uncertainty in atomic forces.
filename: Name of the trajectory file to save sampled configurations.
sampling: Interval at which configurations are sampled and saved.
dt: Timestep for the simulation in femtoseconds.
tau_t: Relaxation time for temperature coupling in femtoseconds.
```

#### Output

The simulation outputs a trajectory file (`trajectory.xyz`) containing sampled atomic configurations and logs (`log.*.uncerA_*`) with energy and uncertainty information.

#### Contributing

Contributions are welcome! For major changes, please open an issue first to discuss what you would like to change.
