from ase.md.langevin import Langevin
from ase.md.nvtberendsen import NVTBerendsen
from ase.md.andersen import Andersen
from ase.md.npt import NPT
from ase import units
from ase.md.velocitydistribution import MaxwellBoltzmannDistribution
from ase.io import read, write, Trajectory
import math
import torch
import numpy

from PaiNN_md import PainnModel
from calculator import MLCalculator_schnet

steps = 0  # Initialize the global variable `steps` to zero.
sampled = 0  # Initialize the global variable `sampled` to zero.

def UDD_MD_run(atoms, Runtime, PATH, Temp, sigma_cutoff, filename, sampling=20, dt=0.5, tau_t=20, intervals=1, A=1000, B=10, C=10, D=10, log_dir="./Results/"):
    """
    Function to perform Molecular Dynamics simulation using UDD (Uncertainty-Driven Dynamics) method.
    
    Parameters:
    atoms (ASE Atoms object): The atoms object containing the initial atomic configuration.
    Runtime (int): Total runtime of the simulation in timesteps.
    PATH (str): Path to the directory containing the trained model weights.
    Temp (float): Temperature of the system in Kelvin.
    sigma_cutoff (float): Cutoff value for uncertainty in the atomic forces.
    filename (str): Name of the trajectory file to save the sampled configurations.
    sampling (int, optional): Interval at which configurations are sampled and saved. Default is 20.
    dt (float, optional): Timestep for the simulation in femtoseconds. Default is 0.5 fs.
    tau_t (float, optional): Relaxation time for temperature coupling in femtoseconds. Default is 20 fs.
    intervals (int, optional): Interval for printing energy and sampling configurations. Default is 1.
    A, B, C, D (float, optional): Parameters for the PainnModel. Default values are 1000, 10, 10, 10 respectively.
    log_dir (str, optional): Directory to save log files. Default is "./Results/".

    Returns:
    int: Total number of simulation steps performed.
    """

    numpy.random.seed(123)  # Seed the random number generator for reproducibility.

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")  # Check if CUDA is available.
    PATH = PATH  # Set the model weights directory.
    weights_name = PATH  # Set the path to the model weights file.
    weights = torch.load(weights_name, map_location=device)  # Load the model weights.
    savedir = log_dir  # Set the directory to save logs and trajectory.

    # Initialize the PainnModel with specified parameters and load weights onto device.
    model = PainnModel(num_interactions=3, hidden_state_size=128, cutoff=5.0, pdb=True, A=A, B=B, C=C, D=D, MD=True)
    model.to(device)
    model.load_state_dict(weights['model_state_dict'], strict=False)

    atoms = atoms  # Set the atoms object to the input parameter.
    mlcalc = MLCalculator_schnet(model)  # Create a calculator using the trained model.
    atoms.calc = mlcalc  # Set the calculator for the atoms object.
    global sampled  # Declare `sampled` as a global variable.
    sampled = 0  # Initialize the sampled counter to zero.
    global steps  # Declare `steps` as a global variable.
    steps = 0  # Initialize the steps counter to zero.

    def printenergy(a=atoms):
        """
        Function to print the potential, kinetic, and total energy of the system.

        Parameters:
        a (ASE Atoms object, optional): The atoms object containing the current atomic configuration.

        Notes:
        This function calculates and prints the potential energy, kinetic energy, forces, and temperature.
        """
        uncer_mean = a.calc.get_property('uncertainty_mean')
        uncer_std = a.calc.get_property('uncertainty_std')
        ekin = a.get_kinetic_energy()
        for_b_max = a.calc.get_property('forces_biased_max')
        for_b_m = a.calc.get_property('forces_biased_mean')
        for_b_s = a.calc.get_property('forces_biased_std')
        temp = ekin / (1.5 * units.kB) / a.get_global_number_of_atoms()

        if temp > 5000 or math.isnan(temp):
            exit()  # Exit the program if temperature exceeds 5000 K or is NaN.

        global steps  # Access the global `steps` variable.
        steps += intervals  # Increment `steps` by the specified interval.
        with open(savedir + "log." + ".uncerA_" + str(A) + str(Temp), 'a') as f:
            f.write(
                f"Steps={steps:8.0f} Uncer={uncer_mean:8.2f} {uncer_std:8.2f} Ekin={ekin:8.2f} force_bias={for_b_max:8.2f} {for_b_m:8.2f} {for_b_s:8.2f} temperature={temp:8.2f}\n")

    def sample(a=atoms, sigma_cutoff=sigma_cutoff, filename=filename):
        """
        Function to sample and save atomic configurations based on uncertainty criteria.

        Parameters:
        a (ASE Atoms object, optional): The atoms object containing the current atomic configuration.
        sigma_cutoff (float): Cutoff value for uncertainty in the atomic forces.
        filename (str): Name of the trajectory file to save the sampled configurations.
        """
        uncer_mean = a.calc.get_property('uncertainty_mean')
        uncer_std = a.calc.get_property('uncertainty_std')
        
        if uncer_mean > sigma_cutoff or uncer_std > 1.5 * sigma_cutoff:
            global sampled  # Access the global `sampled` variable.
            if steps - sampled > sampling:  # Check if enough steps have passed to sample.
                print(sampled)
                write(filename, a, append=True)  # Write the current atomic configuration to the trajectory file.
                sampled = steps  # Update the `sampled` counter to the current step count.

    MaxwellBoltzmannDistribution(atoms, temperature_K=Temp, rng=numpy.random)  # Initialize atomic velocities.

    # Set up NVT Berendsen dynamics with specified parameters.
    dyn = NVTBerendsen(atoms, dt * units.fs, temperature_K=Temp, taut=tau_t * units.fs)
    dyn.attach(printenergy, interval=intervals)  # Attach energy printing function.
    dyn.attach(sample, interval=intervals)  # Attach sampling function.
    dyn.run(Runtime)  # Run the molecular dynamics simulation for specified runtime.

    return steps  # Return the total number of steps performed in the simulation.
