import numpy as np
from time import time
import debugger_module as dbg
from ising_module import hamiltonian_ising

lambdas = [-1]
N_values = [12,13,14,15]
for num_sites in N_values:
    for ll in lambdas:   
        print(f'lambda={ll}, N={num_sites}')

        tic=time()
        H_isi = hamiltonian_ising(N=num_sites, field_strength=ll)
        eigvals, eigvecs = np.linalg.eigh(H_isi)
        tac=time()
        print(f'OK ({tac-tic} seconds)\n')

### RISULTATI ###
# NON HO NOTATO DIFFERENZE NEL CPUTIME VARIANDO LAMMDA (TESTATO FINO A N=12)
# N=12, lambda=-1 ci mette fra i 20 e i 45 secondi
# N=13 crasha