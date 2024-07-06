from ase.io import read
from udd_run import UDD_MD_run
steps=0
atoms = read("/PATH/TO/INITIAL/CONFIGURATION")
A = 2000
B = 20
C = 20
D = 7
a = UDD_MD_run(atoms, Runtime=100000, PATH="/PATH/TO/MODEL.pt", Temp=300, sigma_cutoff=55, filename="/PATH/TO/SAMPLED/TRAJECTORY", sampling=500, dt=0.1, tau_t=10, intervals=1, A=A, B=B, C=C, D=D, log_dir="/PATH/TO/LOG/DIR")
