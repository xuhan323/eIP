Uncertainty-Driven Dynamics (UDD) Molecular Dynamics Simulation
This repository contains code for performing Molecular Dynamics (MD) simulations using the Uncertainty-Driven Dynamics (UDD) method, integrated with machine learning potentials.

Overview
The UDD method enhances traditional MD simulations by leveraging uncertainty estimates from machine learning models to dynamically adjust simulation parameters. This implementation utilizes a neural network potential (PaiNN) to compute atomic forces and uncertainties during the simulation.

Requirements
Python 3.x
ASE (Atomic Simulation Environment)
PyTorch
NumPy
Installation
Clone the repository:

bash

git clone https://github.com/your_username/udd-md-simulation.git
cd udd-md-simulation
Install dependencies:

pip install -r requirements.txt

Usage
Running a Simulation
To run a UDD MD simulation, use the UDD_MD_run function defined in udd_md_simulation.py. Modify the parameters as needed:

Python example:

    from udd_md_simulation import UDD_MD_run
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
    Parameters
    atoms: ASE Atoms object containing the initial atomic configuration.
    Runtime: Total runtime of the simulation in femtoseconds.
    PATH: Path to the directory containing the trained model weights.
    Temp: Temperature of the system in Kelvin.
    sigma_cutoff: Cutoff value for uncertainty in atomic forces.
    filename: Name of the trajectory file to save sampled configurations.
    sampling: Interval at which configurations are sampled and saved.
    dt: Timestep for the simulation in femtoseconds.
    tau_t: Relaxation time for temperature coupling in femtoseconds.

Output
The simulation outputs a trajectory file (trajectory.xyz) containing sampled atomic configurations and logs (log.*.uncerA_*) with energy and uncertainty information.

Contributing
Contributions are welcome! For major changes, please open an issue first to discuss what you would like to change.