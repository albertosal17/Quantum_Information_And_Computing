import numpy as np
import scipy.sparse as sprs

import debugger_module as dbg
from N_body_density import get_reduced_density_matrix

def pauli_matrix(choice):
    """
    !! Generate a Pauli matrix in sparse CSR format.
    !! 
    !! This function returns one of the three Pauli matrices ('x', 'y', or 'z') 
    !! as a sparse matrix in CSR format. Pauli matrices are fundamental in quantum mechanics
    !! and are used for describing spin-1/2 systems and qubits.
    !!
    !! @param choice [in] A string indicating the desired Pauli matrix:
    !!                     - 'x': Pauli-X matrix
    !!                     - 'y': Pauli-Y matrix
    !!                     - 'z': Pauli-Z matrix
    !! 
    !! @return A scipy.sparse.csr_matrix object representing the requested Pauli matrix.
    !!
    !! @throws Raises an error if an invalid `choice` argument is passed.
    """

    if choice=='x':
        return sprs.csr_matrix(np.array([[0,1],
                                         [1,0]]))
    elif choice=='y':
        return sprs.csr_matrix(np.array([[0,-1j],
                                         [1j,0]]))
    elif choice=='z':
        return sprs.csr_matrix(np.array([[1,0],
                                         [0,-1]]))
    else:
        dbg.error(f"Invalid input argument for 'choice': should be 'x', 'y', or 'z', while {choice} was passed")



def gen_hamiltonian_field(N, debug=False):
    """
    !! Generate the field contribution to the Hamiltonian for quantum Ising model in transverse field.
    !!
    !! This function computes the contribution of an external field to the Hamiltonian 
    !! for a system of `N` spins. The field interacts with each spin via the Pauli-Z operator.
    !! 
    !! @param N [in] The number of spins in the system. Must be at least 2.
    !! @param debug [in] Boolean flag to enable or disable debug checkpoints.
    !!
    !! @return A scipy.sparse.csr_matrix representing the field contribution to the Hamiltonian.
    !!
    !! @throws Raises an error if `N` is less than 2, or if unexpected conditions are encountered during execution.
    """

    # Validate the input: a many-body system requires at least 2 spins
    if N < 2:
        dbg.error("N should be at least 2 for a many-body system")

    # List to store individual contributions for each spin
    single_spin_contributes = [] 
    # Loop over each spin in the system (from 1 to N inclusive)
    for kk in np.arange(1, N + 1):  
        matrices = []  # List to store matrices involved in the tensor product for this spin

        dbg.checkpoint('---------------', debug=debug)
        dbg.checkpoint(f'index: {kk}', debug=debug)

        # Construct the left-side identity matrix for the tensor product
        if (kk - 1) != 0:  # Only add if the spin is not the first one
            mat_sx = sprs.identity(2**(kk - 1)) 
            dbg.checkpoint(f'mat_sx\n{mat_sx}', debug=debug)
            matrices.append(mat_sx) 

        # Add the Pauli-Z matrix for the interaction of the spin at position kk
        sigma_z = pauli_matrix('z')  
        matrices.append(sigma_z)

        # Construct the right-side identity matrix for the tensor product
        if N - kk != 0:  # Only add if the spin is not the last one
            mat_dx = sprs.identity(2**(N - kk))  
            dbg.checkpoint(f'mat_dx\n{mat_dx}', debug=debug)
            matrices.append(mat_dx)  

        # Compute the tensor product of all matrices for this spin
        product = sprs.kron(matrices[0], matrices[1])  # Combine the first two matrices
        if len(matrices) == 3:  # If there are three subterms for the tensor product: identity, sigma_z, identity
            product = sprs.kron(product, matrices[2]) 
        elif len(matrices) > 3:  # Sanity check: there should not be more than 3 matrices
            dbg.error("Something unexpected happened: 'matrices' should have at most 3 elements (left, middle, right)")

        # Append the tensor product contribution for this spin
        single_spin_contributes.append(product)

    # Sum all single-spin contributions to form the field contribution to the Hamiltonian
    if not len(single_spin_contributes) == N:  # Sanity check: ensure contributions match the number of spins
        dbg.error("Something unexpected happened: the number of contributions should equal the number of spins")
    
    # Initialize the sum with the first contribution
    field_contribute_to_H = single_spin_contributes[0]
    # Add all other contributions
    for jj in np.arange(1, N):
        field_contribute_to_H += single_spin_contributes[jj]

    return field_contribute_to_H

def gen_hamiltonian_pairs(N, debug=False):
    """
    !! Generate the pair interaction contribution to the Hamiltonian for quantum Ising model in transverse field.
    !!
    !! This function computes the contribution of pairwise spin interactions (e.g., Ising-type interaction)
    !! to the Hamiltonian for a system of `N` spins. The interaction is modeled using the Pauli-X operator
    !! acting on neighboring pairs of spins.
    !! 
    !! @param N [in] The number of spins in the system. Must be at least 2.
    !! @param debug [in] Boolean flag to enable or disable debug checkpoints.
    !!
    !! @return A scipy.sparse.csr_matrix representing the pair interaction contribution to the Hamiltonian.
    !!
    !! @throws Raises an error if `N` is less than 2 or if unexpected conditions are encountered during execution.
    """
    # Validate input: a many-body system requires at least 2 spins
    if N < 2:
        dbg.error("N should be at least 2 for a many-body system")

    # List to store the Hamiltonian contributions for each pair of spins
    single_pair_contributes = []

    # Loop over each pair of neighboring spins (from spin 1 to N-1)
    for ii in np.arange(1, N): 
        matricess = []  # List to store matrices for tensor product construction

        dbg.checkpoint('---------------', debug=debug)
        dbg.checkpoint(f'index: {ii}', debug=debug)

        # Construct the left-side identity matrix for the tensor product
        if (ii - 1) != 0:  # Add only if the pair is not at the start of the chain
            mat_sx = sprs.identity(2**(ii - 1)) 
            dbg.checkpoint(f'mat_sx\n{mat_sx}', debug=debug)
            matricess.append(mat_sx)

        # Add the Pauli-X operators for the pair of spins
        sigma_x_i = pauli_matrix('x')  # Pauli-X for the first spin in the pair
        sigma_x_i_plus_one = pauli_matrix('x')  # Pauli-X for the second spin in the pair
        matricess.append(sigma_x_i)
        matricess.append(sigma_x_i_plus_one)

        # Construct the right-side identity matrix for the tensor product
        if N - (ii + 1) != 0:  # Add only if the pair is not at the end of the chain
            mat_dx = sprs.identity(2**(N - (ii + 1)))  
            dbg.checkpoint(f'mat_dx\n{mat_dx}', debug=debug)
            matricess.append(mat_dx)

        # Compute the tensor product of all matrices for this pair
        productt = sprs.kron(matricess[0], matricess[1])  # Combine the first two matrices
        productt = sprs.kron(productt, matricess[2])  # Combine with the third matrix
        if len(matricess) == 4:  # If there is a fourth matrix, combine it
            productt = sprs.kron(productt, matricess[3])  # Combine with the fourth matrix
        elif len(matricess) > 4:  # Sanity check: there should not be more than 4 matrices
            dbg.error("Something unexpected happened: 'matricess' should have at most 4 elements (left, middle, right)")

        # Append the tensor product contribution for this pair of spins
        single_pair_contributes.append(productt)

    # Ensure the number of contributions matches the number of pairs
    if not len(single_pair_contributes) == (N - 1):
        dbg.error("Something unexpected happened: we should have a number of terms to sum equal to N-1 pairs")

    # Sum all pair contributions to form the pair interaction contribution to the Hamiltonian
    pairs_contribute_to_H = single_pair_contributes[0]
    for hh in np.arange(1, N - 1):
        pairs_contribute_to_H += single_pair_contributes[hh]

    return pairs_contribute_to_H


def hamiltonian_ising(N, field_strength, ferromagnetic = False, debug=False, print_hamiltonian=False):
    """
    !! Generate the Ising model Hamiltonian for the quantum Ising model in tranverse field.
    !!
    !! The Hamiltonian includes contributions from an external field acting on individual spins and pairwise
    !! interactions between neighboring spins. The sign of the pair interaction term depends on whether
    !! the system is ferromagnetic or antiferromagnetic. 
    !!
    !! @param N [in] The number of spins in the system. Must be at least 2.
    !! @param field_strength [in] The strength of the external field.
    !! @param ferromagnetic [in] Boolean flag indicating whether the system is ferromagnetic (default: False).
    !! @param debug [in] Boolean flag to enable or disable debug checkpoints.
    !! @param print_hamiltonian [in] Boolean flag to print the full Hamiltonian matrix for inspection (default: False).
    !!
    !! @return A scipy.sparse.csr_matrix representing the full Ising model Hamiltonian.
    !!
    !! @throws Raises an error if `N` is less than 2 or if unexpected conditions are encountered during execution.
    """   
    # Generate the contribution to the Hamiltonian from the external field
    H_field = gen_hamiltonian_field(N, debug)
    
    # Generate the contribution to the Hamiltonian from pairwise spin interactions
    H_pairs = gen_hamiltonian_pairs(N, debug)
    
    # Combine field and pair contributions to form the full Hamiltonian
    if ferromagnetic:
        # Ferromagnetic interaction: coupling constant J=+1
        H = field_strength * H_field - H_pairs
    else:
        # Antiferromagnetic interaction: coupling constant J=+1
        H = field_strength * H_field + H_pairs

    dbg.checkpoint(H.shape, debug=debug)

    # Optionally print the full Hamiltonian matrix
    if print_hamiltonian:
        # Set NumPy print options for better formatting of the output
        np.set_printoptions(
            precision=2,      # Number of decimal places
            suppress=True,     # Avoid scientific notation for small numbers
            linewidth=120      # Total width of the output line
        )
        print(H)

    return H


def magnetization_z(state, N, debug=False):
    """
    !! Compute the expectation value of the magnetization operator along the z-axis.
    !!
    !! @param state [in] State vector of the quantum system, represented as a 1D numpy array in the wave function notation.
    !! @param N [in] Number of spins in the system.
    !! @param debug [in] Boolean flag to enable or disable debug checkpoints (default: False).
    !!
    !! @return The expectation value of the magnetization operator along the z-axis.
    !!
    !! @throws Raises an error if the input state is not normalized.
    """

    # Generate the magnetization operator along the z direction (normalized by N)
    # Oss. Since it coincides witht he field contribution to the Hamiltonian, we can reuse the function
    M_z = gen_hamiltonian_field(N) / N

    # Check if the input state is normalized
    norm = np.dot(np.conj(state.T), state)
    if not np.isclose(norm, 1):
        dbg.error(f"The input state is not normalized: norm={norm}")
    else:
        dbg.checkpoint("The input state is normalized", debug=debug)

    # Compute the expectation value of M_z
    A = M_z.dot(state)  # Apply M_z to the state
    B = np.dot(np.conj(state.T), A)  # Inner product with the state
    exp_value_M_z = B

    return exp_value_M_z


def magnetization_x(state, N, debug=False):
    """
    !! Compute the expectation value of the magnetization operator along the x-axis.
    !!
    !! @param state [in] State vector of the quantum system, represented as a 1D numpy array.
    !! @param N [in] Number of spins in the system.
    !! @param debug [in] Boolean flag to enable or disable debug checkpoints (default: False).
    !!
    !! @return The expectation value of the magnetization operator along the x-axis.
    !!
    !! @throws Raises an error if the input state is not normalized.
"""

    # Generating the magnetization operator along the x direction
    single_spin_contributes = []
    for kk in np.arange(1, N + 1):  # From 1 to N
        matrices = []

        dbg.checkpoint('---------------', debug=debug)
        dbg.checkpoint(f'index: {kk}', debug=debug)

        # Left-side tensor product
        if (kk - 1) != 0:  # Only for indices that are not the first one
            mat_sx = sprs.identity(2 ** (kk - 1))
            dbg.checkpoint(f'mat_sx\n{mat_sx}', debug=debug)
            matrices.append(mat_sx)

        # Pauli matrix for spin at position kk
        sigma_x = pauli_matrix('x')
        matrices.append(sigma_x)

        # Right-side tensor product
        if N - kk != 0:  # Only for indices that are not the last one
            mat_dx = sprs.identity(2 ** (N - kk))
            dbg.checkpoint(f'mat_dx\n{mat_dx}', debug=debug)
            matrices.append(mat_dx)

        # Compute the tensor product
        product = sprs.kron(matrices[0], matrices[1])
        if len(matrices) == 3:  # If there are 3 components
            product = sprs.kron(product, matrices[2])
        elif len(matrices) > 3:
            dbg.error("Unexpected condition: 'product' should have at most 3 elements.")

        single_spin_contributes.append(product)

    # Sum all single spin contributions
    if not len(single_spin_contributes) == N:
        dbg.error("Unexpected condition: contributions should equal the number of spins.")
    M_x = single_spin_contributes[0]
    for jj in np.arange(1, N):
        M_x += single_spin_contributes[jj]

    M_x = M_x / N  # Normalize by the number of spins

    #########################################################################
    # Check if the input state is normalized
    norm = np.dot(np.conj(state.T), state)
    if not np.isclose(norm, 1):
        dbg.error(f"The input state is not normalized: norm={norm}")
    else:
        dbg.checkpoint("The input state is normalized", debug=debug)

    #########################################################################
    # Compute the expectation value of M_x
    A = M_x.dot(state)
    B = np.dot(np.conj(state.T), A)
    exp_value_M_x = B

    return exp_value_M_x

  

def von_neuman_entropy(psi, loc_dim, n_sites, N_subsysA=1, debug=False):
    """
    Compute the von Neumann entropy of the reduced density matrix of the subsystem composed by
    the first particle in the N-body system.

    Parameters
    ----------
    psi : ndarray
        State vector of the Quantum Many-Body (QMB) system, represented as a 1D numpy array.
    N_subsysA : int
        Number of lattice sites in the subsystem A, counting from the left-side of the total system.
        All other sites compose the second partition. Complessively the system is bi-partized
    loc_dim : int
        Local dimension of each individual site in the QMB system (e.g., for qubits, loc_dim = 2).
    n_sites : int
        Total number of lattice sites in the QMB system.
    debug : bool, optional
        If True, the function will print debug messages at key points. Default is False.

    Returns
    -------
    float
        von Neumann entropy of the quantum state.
    """
    #check that N_subsysA is a valid input: less than the total number of sites
    if N_subsysA >= n_sites:
        dbg.error("N_subsysA must be smaller than the total number of sites.")

    # Compute the reduced density matrix of the subsystem A
    rho_A = get_reduced_density_matrix(psi, loc_dim, n_sites, keep_indices=list(range(N_subsysA)), debug=debug)

    # Compute the eigenvalues of the reduced density matrix
    eigvals = np.linalg.eigvalsh(rho_A)

    # Compute the von Neumann entropy (in the eigen basis wrt rho_A is diagonal) 
    S = -np.sum(eigvals*np.log2(eigvals+ 1*10**-12)) # -tr(rho_A*log2(rho_A))

    return S



    
