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

# How to run this code:

### Notebook (Demo)

In `eIP.ipynb`,  we have demonstrated the training process of the eIP model using the small molecule dataset as an example, as described in the article. The modifications to the parameters and the selection of datasets involved in the article are as follows:

By modifying the test.py, you can achieve changes to the hyperparameters in eIP.

### UDD

To run a UDD MD simulation, use the UDD_MD_run function defined in udd_md_simulation.py. Modify the parameters as needed:

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

The simulation outputs a trajectory file (trajectory.xyz) containing sampled atomic configurations and logs (log.*.uncerA_*) with energy and uncertainty information.

#### Contributing

Contributions are welcome! For major changes, please open an issue first to discuss what you would like to change.

