from ase.io import read
from udd_run import UDD_MD_run
from ase.io.lammpsdata import read_lammps_data
from ase.io.trajectory import Trajectory

# 读取轨迹文件

steps=0
atoms = read("/home/tong/AIBMD2Drug-taoyong/AlphaNet/test/silica/silica_uncertain/LiFePO4.cif", format='cif')



num_atoms = len(atoms)
print(f"总原子数：{num_atoms}")
A = 4500
B = 10
C = 5
D = 200
a = UDD_MD_run(atoms, 
               Runtime=5000000, 
               PATH="./checkpoint/checkpoint_general.pt",
               Temp=300, sigma_cutoff=55, 
               filename=f"PE_datalog/A_{A}_B{B}_C{C}_D{D}.traj", 
               sampling=50, dt=0.1, tau_t=10, intervals=1, A=A, B=B, C=C, D=D, 
               log_dir="PE_datalog/"
               )
