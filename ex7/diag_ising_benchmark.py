import numpy as np
from time import time
import scipy.sparse as sprs
import pandas as pd

import debugger_module as dbg
from ising_module_sparse import hamiltonian_ising


# Define parameters
lambdas = [-3,-2,-1,0]  # List of values for the external field strength 
N_values = [10,12,14]  # List of system sizes (number of sites)
nr_eigs = 10  # Number of eigenvalues to compute

debug = False  # Debug flag for conditional debugging messages

# Loop over system sizes and lambda values
for num_sites in N_values:
    for ll in lambdas:
        print(f'lambda={ll}, N={num_sites}')  

        tic = time()  
        # Compute the Ising Hamiltonian for given parameters
        H_isi = hamiltonian_ising(N=num_sites, field_strength=ll)

        # Check if the Hamiltonian is Hermitian
        difference = H_isi - H_isi.getH()  # Difference between H and its conjugate transpose
        if not np.all(np.abs(difference.data) < 1e-10):  # Verify that the difference is negligible
            dbg.error("The input matrix is not hermitian")  

        # Diagonalize the Hamiltonian to find eigenvalues and eigenvectors
        eigvals, eigvecs = sprs.linalg.eigsh(H_isi, k=nr_eigs, which='SA')  # Compute smallest eigenvalues
        tac = time()  
        print(f'OK ({tac-tic} seconds)\n')  

        # Create a dataframe to store eigenvalues and parameters
        df = pd.DataFrame({
            "Level": np.arange(eigvals.shape[0]),  # Energy level indices
            "Eigenvalues": eigvals / num_sites,  # Normalized eigenvalues
            "lambda": np.repeat(ll, repeats=eigvals.shape[0]),  # Lambda value
            "N": np.repeat(num_sites, repeats=eigvals.shape[0]),  # Number of sites
        })

        # Save the dataframe to a CSV file
        df.to_csv(
            f'./data/N{num_sites}_lambda{ll}.csv',  # File path
            index=False,  # Do not write row indices
            header=False,  # Do not include headers in the CSV
            mode='w'  # Overwrite existing file
        )
        
        dbg.checkpoint("Eigenvalues saved", debug=debug)