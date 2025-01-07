import numpy as np
import scipy.sparse as sprs
import pandas as pd
import matplotlib.pyplot as plt

import debugger_module as dbg
from ising_module_sparse import hamiltonian_ising, magnetization_z, magnetization_x, von_neuman_entropy

# Define the range of lambda values using a logarithmic scale
lambda_values = -np.logspace(-3,2,30)

# Set the parameters of the simulation the system
num_sites = 6
debug=True
ferromagnetic=True #ferromagnetic interactions between spins
loc_dim=2 #we are considering a QMB system made of spin 1/2 particles
k=1 #number of eigenvalues to compute

# Initialize lists to store results for each value of lambda
mz_values, mx_values = [], []
entangs = []
for ll in lambda_values: 
    dbg.checkpoint(f'lambda={ll}', debug=debug)
    # compute hamiltonian
    H_isi = hamiltonian_ising(N=num_sites, field_strength=ll, debug=False, ferromagnetic=ferromagnetic) #ATTTENZIONE LA STO MODIFICANDO

    # check hermitian
    difference = H_isi - H_isi.getH()
    if not np.all(np.abs(difference.data) < 1e-10):
        dbg.error("The input matrix is not hermitian")

    # diagonalize hamiltonian
    eigvals, eigvecs = sprs.linalg.eigsh(H_isi, k=k, which='SA')  
    dbg.checkpoint('matrix diagonalized', debug=debug)

    #retrieve groundstate
    gs = eigvecs[:,0]

    #compute magnetizations of the groundstate
    mz = magnetization_z(gs, num_sites, debug=False)
    mz_values.append(mz)
    mx = magnetization_x(gs, num_sites, debug=False)
    mx_values.append(mx)

    #compute entanglement between two partitions of the whole system (see doc of the function)
    ent = von_neuman_entropy(psi=gs, loc_dim=loc_dim, n_sites=num_sites, N_subsysA=2, debug=False)
    entangs.append(ent)

#store result
df = pd.DataFrame({ 
    "lambda" : lambda_values, 
    "M_z" : mz_values,
    "M_x" : mx_values,
    "Entanglement" : entangs
    })
df.to_csv(f'./data/magnetization_N{num_sites}.csv', index=False, header= False, mode='w') #write mode: 'w' to overwrite the file
dbg.checkpoint("Result saved", debug=debug)


