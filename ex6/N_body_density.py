from time import time
from math import prod
import numpy as np
import matplotlib.pyplot as plt

import debugger_module as dbg 

# ASSIGNMENT 2
#   1) Given N=2, write the density matrix of a general pure state Ψ, 𝜌 = |Ψ⟩⟨Ψ|
#
#   2) Given a generic density matrix of dimension 𝐷**𝑁 x 𝐷**𝑁 compute the reduced density matrix of
#   either the left or the right system, e.g. 𝜌1 = Tr2𝜌.
#
#   3) Test the functions described before (and all others needed) on two-spin one-half (qubits) with different states

def density_matrix(state):
    return np.outer(state,state.conj())

def get_reduced_density_matrix(psi, loc_dim, n_sites, keep_indices,
    print_rho=False, debug=False):
    """
    Compute the reduced density matrix of a subsystem by tracing out the environment.
    
    Parameters
    ----------
    psi : ndarray
        State vector of the Quantum Many-Body (QMB) system, represented as a 1D numpy array.
    loc_dim : int
        Local dimension of each individual site in the QMB system (e.g., for qubits, loc_dim = 2).
    n_sites : int
        Total number of lattice sites in the QMB system.
    keep_indices : list of ints
        Indices of the lattice sites that are part of the subsystem to be kept.
            **Note**: `keep_indices` specifies which sites of the system should be included in the reduced density matrix. 
            The sites not included in this list will be traced out (i.e., considered part of the environment).
            For example, if `keep_indices=[1,2]`, it means we want to keep the second and third sites of the system 
            as the subsystem, and the remaining sites will be traced out to compute the reduced density matrix. 
    print_rho : bool, optional
        If True, the function will print the resulting reduced density matrix. Default is False.
    debug : bool, optional
        If True, the function will print debug messages at key points. Default is False.

    Returns
    -------
    ndarray
        Reduced density matrix (RDM) of the subsystem.
    """
    # Security checks for input types
    if not isinstance(psi, np.ndarray):
        raise TypeError(f'density_mat should be an ndarray, not a {type(psi)}')

    if not np.isscalar(loc_dim) and not isinstance(loc_dim, int):
        raise TypeError(f'loc_dim must be an SCALAR & INTEGER, not a {type(loc_dim)}')

    if not np.isscalar(n_sites) and not isinstance(n_sites, int):
        raise TypeError(f'n_sites must be an SCALAR & INTEGER, not a {type(n_sites)}')

    # Ensure psi is reshaped into a tensor with one leg per lattice site
    psi_tensor = psi.reshape(*np.repeat(loc_dim, n_sites))

    # Determine which indices correspond to the environment (to be traced out)
    all_indices = list(range(n_sites))
    env_indices = [i for i in all_indices if i not in keep_indices] #sites of the lattice to be tracced out
    new_order = keep_indices + env_indices #concatenation of lists through operator +

    # Reorder the tensor to group subsystem and environment indices
    psi_tensor = np.transpose(psi_tensor, axes=new_order)
    dbg.checkpoint(f"Reordered psi_tensor shape: {psi_tensor.shape}", debug=debug)

    # Calculate dimensions of the subsystem and the environment
    subsystem_dim = np.prod([loc_dim for i in keep_indices])
    env_dim = np.prod([loc_dim for i in env_indices])

    # Reshape tensor to separate the subsystem and environment
    psi_partitioned = psi_tensor.reshape((subsystem_dim, env_dim))

    # Compute the reduced density matrix by tracing out the environment
    RDM = np.tensordot(psi_partitioned, np.conjugate(psi_partitioned), axes=([1], [1])) # the axes=([1], [1]) argument tells np.tensordot() to contract over the second axis (axis 1) of psi_partitioned (the env_dim dimension). 

    # Reshape rho to ensure it is a square matrix corresponding to the subsystem
    RDM = RDM.reshape((subsystem_dim, subsystem_dim))

    # Optionally print the reduced density matrix
    if print_rho:
        print('----------------------------------------------------')
        print(f'DENSITY MATRIX TRACING SITES ({str(env_indices)})')
        print('----------------------------------------------------')
        print(RDM)

    return RDM



##Esempio di utilizzo
# N=3
# D=2

# state = generate_Nbody_wfc(D,N) #controlla che questo sia un pure state
# print(state)

# density_mat = density_matrix(state) #FORMULA DELLA MATRICE DI DENSITA'
# print(density_mat)

# get_reduced_density_matrix(state, D, N, keep_indices=[1,2])