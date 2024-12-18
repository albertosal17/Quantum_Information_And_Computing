import numpy as np
from math import isclose

import debugger_module as dbg


def density_matrix_pauli(r_x, r_y, r_z):
    """
    Generate a density matrix expressed in the complete basis formed by the pauli matrices plus the identity
    
    Parameters
    ----------
    r_x : float
        Coefficient for the Pauli-X matrix (sigma_x).
    r_y : float
        Coefficient for the Pauli-Y matrix (sigma_y).
    r_z : float
        Coefficient for the Pauli-Z matrix (sigma_z).
    
    Returns
    -------
    ndarray
        A 2x2 matrix representing the generalized Pauli matrix.
    """
    #PAULI MATRICES
    sigma_x = np.array([[0,1],
                    [1,0]])
    sigma_y = np.array([[0,-1j],
                        [1j,0]])
    sigma_z = np.array([[1,0],
                        [0,-1]])
    
    return 0.5 * (np.eye(2) + r_x*sigma_x + r_y*sigma_y + r_z*sigma_z) 


def qubit_bloch_vector(dens_matr):
    """
    Compute the Bloch vector of a single-qubit density matrix.

    Parameters
    ----------
    dens_matr : ndarray
        2x2 density matrix representing the qubit state.
    
    Returns
    -------
    ndarray
        Bloch vector [x, y, z], where:
        x = 2 * Re(dens_matr[1, 0]),
        y = 2 * Im(dens_matr[1, 0]),
        z = 2 * dens_matr[0, 0] - 1.

    Examples
    --------
    |0⟩ state: [[1, 0], [0, 0]] -> [0, 0, 1]
    |1⟩ state: [[0, 0], [0, 1]] -> [0, 0, -1]
    """
    a = dens_matr[0, 0]
    b = dens_matr[1, 0]
    x = np.real(2.0 * b.real)
    y = np.real(2.0 * b.imag)
    z = np.real(2.0 * a - 1.0)
    
    bloch_v = np.array([x,y,z])

    #POSSIBILE UPDATE: RITORNARE ANCHE LA NORMA DEL VETTORE
    return bloch_v

def state_purity(d_mat):
    """
    Check if a quantum state is pure or mixed based on its density matrix.

    Parameters
    ----------
    d_mat : ndarray
        Density matrix of the quantum state.

    Returns
    -------
    bool
        True if the state is pure (purity ≈ 1), False if it is mixed.
    """
    purity = np.trace(np.dot(d_mat, d_mat))
    
    if isclose(np.real(purity), 1, rel_tol=1e-6):
        print(f"State is pure")
        return True
    else:
        print(f"Mixed state with purity {np.round(np.real(purity),4)}")
        return False


def expectation_value(observable, wfc=None, rho=None, debug=False):
    """
    Compute the expectation value of an observable for a given quantum state.

    Parameters
    ----------
    observable : ndarray
        Hermitian matrix representing the observable.
    wfc : ndarray, optional
        1D array representing the wavefunction of the quantum state.
    rho : ndarray, optional
        2D matrix representing the density matrix of the quantum state.
    debug : bool, optional
        If True, prints debug information. Default is False.

    Returns
    -------
    float
        The expectation value of the observable.

    Raises
    ------
    ValueError
        If both `wfc` and `rho` are provided.
    """
    if wfc is not None and rho is not None:
        dbg.error("Only one between the density matrix (rho) and wavefunction (psi) should be passed")    
    
    elif wfc is not None:
        dbg.checkpoint("Wavefunction was passed", debug=debug)
        e_v = np.vdot(wfc, np.dot(observable, wfc)) #np.vdot(psi, ...) computes the conjugate transpose of psi and performs the inner product.
    
    elif rho is not None:
        dbg.checkpoint("Density matrix was passed", debug=debug)
        e_v = np.trace(np.dot(rho,observable))
    
    return np.real(e_v)

def get_reduced_density_matrix_from_density_matrix(rho, subsys_to_trace, display=False):
    density_tens = rho.reshape((2,2,2,2))

    if subsys_to_trace == 0:
        idxs = [0,2]
    else:
        idxs = [1,3]

    rho_A = np.trace(density_tens, axis1=idxs[0], axis2=idxs[1])

    if display:
        print(rho_A)
    
    return rho_A