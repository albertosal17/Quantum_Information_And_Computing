import numpy as np
import matplotlib.pyplot as plt

import debugger_module as dbg 


# ASSIGNMENT 1
#   Write a code for writing the total wavefunction of a N-body composite system
#   CASE 1: non interacting, separable pure state;
#   CASE 2: general 𝑁-body pure wave function Ψ ∈ H^{𝐷𝑁}
#   Compare their efficiency;

def normalize_vector(vv):
    """
    Normalize a complex vector.
    
    Arguments:
        vv (numpy.ndarray): Input vector to be normalized.

    Returns:
        numpy.ndarray: Normalized vector.
    """
    vv_n = vv / np.sqrt(np.sum(np.abs(vv)**2))
    return vv_n


def generate_Nbody_wfc(D, N, coeffs=None, print_final=False, debug=False):
    """ 
    Generate the wavefunction for a general N-body composite system.

    Arguments:
        D (int): Dimension of the Hilbert space of each subsystem.
        N (int): Number of subsystems.
        coeffs (numpy.ndarray, optional): Predefined coefficients for the wavefunction.
                                          If None, random coefficients are generated.
        
    Returns:
        numpy.ndarray: Normalized wavefunction for the general N-body system.
    """
    if coeffs is None:
        # Generate random coefficients for the wavefunction
        vector = np.random.random(D**N) + 1j * np.random.random(D**N) # D**N coefficients required
        vector = normalize_vector(vector)
        dbg.checkpoint("Random state generated", debug=debug)

    else:
        if coeffs.shape[0] != (D**N,):
            dbg.error(f"Wrong input shape for coeffs. Should have shape ({D**N},), while it has shape {coeffs.shape}")
        else:
            vector = normalize_vector(coeffs)
    
    if print_final:
        print(vector)

    return vector

def Nbody_separable_wfc(D, N, compute_total_state=True, subsys_states=None, print_state=False, debug=False):
    """
    Generate the wavefunction for an N-body composite system in a separable pure state.

    Arguments:
        D (int): Dimension of the Hilbert space of each subsystem.
        N (int): Number of subsystems.
        compute_total_state (bool): If True, compute and returns the (normalized) tensor product 
                                    of the states of the subsystems. Default is True.
        subsys_states (numpy.ndarray, optional): Pure states for each subsystem.
                                                 If None, random pure states are generated.
        debug (bool): If True, debugging checkpoints are enabled.

    Returns:
        numpy.ndarray: Normalized separable wavefunction for the N-body system.
    """
    
    # cosa succede se ha dimensione 1
    if subsys_states is None:
        # Generate random pure states for each subsystem
        subsys_states = np.array([np.random.random(D) + 1j * np.random.random(D) for _ in range(N)]) # D*N coefficients required
        subsys_states = np.array([normalize_vector(state) for state in subsys_states])
        dbg.checkpoint("Random states generated", debug=debug)

    else:
        # check if dimension of subsys_states from input is appropriate 
        if subsys_states.shape != (N,D):
            dbg.error(f"Wrong input shape for subsys_states. Should have shape ({N},{D}), while it has shape {subsys_states.shape}")
    
    # Taking the tensor product of each subsystem state
    if compute_total_state:
        total_state = subsys_states[0] 
        for ii in np.arange(start=1, stop=subsys_states.shape[0]):
            total_state = np.kron(total_state, subsys_states[ii]) 
            total_state = normalize_vector(total_state)

        if print_state:
            print(total_state)
        return total_state
    else:
        return subsys_states


## Esempio di utilizzo 
# N = 3 #number of subsystems (e.g. particles)
# D = 2 #dimension of each subsystem (e.g. 2 for a spin 1/2, i.e. "qubit")
# print(Nbody_separable_wfc(D,N, compute_total_state=False, debug=True))
# sub_states = np.array([[0,1],[1,0],[1,0]], ndmin=2)
# sep_Nbody_state = Nbody_separable_wfc(D,N, sub_states, print_state=True, debug=True)
# print('------------------------------')
# general_Nbody_state = generate_Nbody_wfc(D,N, print_final=True, debug=True)


