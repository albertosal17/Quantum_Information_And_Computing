import numpy as np
from time import time
import scipy.sparse as sprs
import pandas as pd

import debugger_module as dbg
from ising_module_sparse import hamiltonian_ising

lambdas = [0,-1,-2,-3]
N_values = [15,16,17,18,19]
nr_eigs=30

debug=False

for num_sites in N_values:
    for ll in lambdas:   
        print(f'lambda={ll}, N={num_sites}')

        tic=time()
        # compute hamiltonian
        H_isi = hamiltonian_ising(N=num_sites, field_strength=ll)
        # check hermitian
        difference = H_isi - H_isi.getH()
        if not np.all(np.abs(difference.data) < 1e-10):
            dbg.error("The input matrix is not hermitian")
        # diagonalize hamiltonian
        eigvals, eigvecs = sprs.linalg.eigsh(H_isi, k=nr_eigs, which='SA')  
        tac=time()
        print(f'OK ({tac-tic} seconds)\n')

        df = pd.DataFrame({
            "Level" : np.arange(eigvals.shape[0]),
            "Eigenvalues": eigvals, 
            "lambda" : np.repeat(ll, repeats=eigvals.shape[0]), 
            "N" : np.repeat(num_sites, repeats=eigvals.shape[0]),
        })
        df.to_csv(f'./data/N{num_sites}_lambda{ll}_{nr_eigs}eigs.csv', index=False, header= False, mode='a') #append mode
        dbg.checkpoint("Eigenvalues saved", debug=debug)


### RISULTATI ###

# 1CASO: SENZA SPARSE 
#   NON HO NOTATO DIFFERENZE NEL CPUTIME VARIANDO LAMMDA (TESTATO FINO A N=12)
#   N=12, lambda=-1 ci mette fra i 20 e i 45 secondi
#   N=13 crasha
#
# 2CASO: SPARSE 
#   Intanto provo con k=10...
#   N=20 (shape (1048576, 1048576)), lambda=-1 ci mette 66 secondi 
#   N=21 crasha
#
#   Variando lambda:
#       lambda=0, N=19
#       OK (8.841381788253784 seconds)

#       lambda=-1, N=19
#       OK (19.57046341896057 seconds)

#       lambda=-2, N=19
#       OK (16.997222900390625 seconds)

#       lambda=-3, N=19
#       OK (21.245192527770996 seconds)

#       Conclusione: con lambda =0 ci mette sempre quasi la metà del tempo. ha senso: ci sono meno entries nella sparse matrix. Nel
#                caso non sparse questo non si apprezzava perchè non variava il numero complessivo di entries nella matrice finale
#
#   Con k=20, N=20, lambda=-1 ci mette 76 secondi


# UN APPUNTO:
# Since the Pauli matrices are Hermitian, and linear combinations of Hermitian operators with real 
# coefficients remain Hermitian, the Hamiltonian HH is Hermitian. Consequently, all eigenvalues of HH are real.