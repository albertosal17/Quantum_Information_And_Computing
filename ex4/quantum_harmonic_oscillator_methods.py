import numpy as np
from math import factorial
from time import time
import matplotlib.pyplot as plt
import itertools
import pandas as pd
from scipy.integrate import simpson

from matrix_methods import diagonalize_hermitian
from matrix_methods import gen_diag_simmetric_mat
from debugger_module import checkpoint, error, warning


color_cycle = plt.cm.Set1.colors  # Setting a predefined colormap for the plots


def hermite(x, n):
    """
    @brief Evaluates the Hermite polynomial \( H_n(x) \) of degree \( n \) at a given point or array of points \( x \).
    
    @param x The input value(s) where the Hermite polynomial should be evaluated. Can be a scalar or a NumPy array.
    @param n The degree of the Hermite polynomial. Must be a non-negative integer.
    
    @return The value(s) of the Hermite polynomial \( H_n(x) \) evaluated at \( x \).
    
    This function constructs the Hermite polynomial \( H_n(x) \) using a coefficient array, where only the \( n \)-th 
    coefficient is set to 1 (all others are zero). The polynomial is then evaluated using NumPy's Hermite polynomial 
    utilities for efficient and numerically stable computation.
    """
    herm_coeffs = np.zeros(n+1)
    herm_coeffs[n] = 1

    return np.polynomial.hermite.hermval(x, herm_coeffs)

def stationary_states(x, omega, m, h_bar, k):
    """
    @brief Computes the first k stationary states (eigenvalues and eigenfunctions) of the quantum harmonic oscillator.
    
    @param x A scalar or NumPy array representing the positions where the eigenfunctions are evaluated.
    @param omega The angular frequency of the harmonic oscillator.
    @param m The mass of the particle.
    @param h_bar Reduced Planck's constant.
    @param k The number of stationary states (eigenstates) to compute.
    
    @return A tuple containing:
        - theoretical_eigvals: A NumPy array of the first \( k \) eigenvalues of the quantum harmonic oscillator.
        - theoretical_eigvecs: A 2D NumPy array where each column corresponds to an eigenfunction evaluated at the given positions \( x \).

    The quantum harmonic oscillator has eigenvalues given by:
        \[
        E_n = \hbar \omega \left( n + \frac{1}{2} \right)
        \]
    and eigenfunctions:
        \[
        \psi_n(x) = \frac{1}{\sqrt{2^n n!}} \left( \frac{m \omega}{\hbar \pi} \right)^{1/4}
        e^{-\frac{m \omega x^2}{2 \hbar}} H_n\left( \sqrt{\frac{m \omega}{\hbar}} x \right)
        \]
    where \( H_n(x) \) are the Hermite polynomials.
    """
    theoretical_eigvals, theoretical_eigfuncs = [], []
    #for each eigenvalue (between the k to be computed)
    for nn in range(k):
        #QHO eigenvalue formula:
        E_n = h_bar * omega * ( nn + 1/2 ) 
        theoretical_eigvals.append(E_n)

        #QHO eigenfunctions formula:
        prefactor_n = ( ( 1. / np.sqrt( 2.**nn * factorial(nn) ) ) * 
                            ( (omega * m) / (h_bar * np.pi) )**(1/4)
        )
        psi_n = ( prefactor_n * 
                    np.exp( -(m * omega * x**2) / (2 * h_bar) ) * 
                    hermite(np.sqrt( (m * omega) / h_bar ) * x, nn) 
        )   
        theoretical_eigfuncs.append(psi_n)      
    theoretical_eigvals, theoretical_eigfuncs = np.array(theoretical_eigvals), np.column_stack(theoretical_eigfuncs)
    
    return theoretical_eigvals, theoretical_eigfuncs

def overlap(v_1, v_2, deltaX):
    """
    @brief Computes the overlap integral between two vectors in a discretized space using Simpson's rule.

    @param v_1 A 1D complex NumPy array representing the first vector.
    @param v_2 A 1D complex NumPy array representing the second vector.
    @param deltaX The spacing between discrete points in the grid (used for the integration).

    @return The absolute value of the overlap integral between the two vectors.

    The overlap integral is computed as:
        \[
        \text{overlap} = \left| \Delta x \sum \left( \psi_1^*(x) \psi_2(x) \right) \right|
        \]
    Using Simpson's rule for numerical integration.

    @note This function assumes that the vectors are of the same dimension and that the vectors are discretized over the same grid spacing \( \Delta x \).
    """
    # Check if the two vectors have the same dimension
    if v_1.shape[0] != v_2.shape[0]:
        error("The vectors have different dimensions!")
    else:
        # Calculate the product of the conjugate of v_1 and v_2
        product = np.conjugate(v_1) * v_2

        # Use scipy's simpson function to integrate the product over the grid points
        overlap = np.abs(simpson(product, dx=deltaX))
        return overlap
    

def QHO_diagonalization(x, k, sparse, order, omega, debug):
    """
    @brief Compute the eigenvalues and eigenvectors of the Quantum Harmonic Oscillator (QHO) Hamiltonian.
    
    This function generates the Hamiltonian matrix of the Quantum Harmonic Oscillator using the finite difference method 
    and diagonalizes it to compute the eigenvalues and eigenvectors. The Hamiltonian consists of a kinetic term and 
    a potential term, both approximated to the desired order of accuracy. The eigenvectors are normalized after diagonalization.

    @param x A 1D numpy array representing the discretized spatial domain (grid points).
    @param k The number of eigenvalues and eigenvectors to compute.
    @param sparse A boolean indicating whether to use sparse matrix representations.
    @param order An integer specifying the approximation order (2 for second-order, 3 for third-order).
    @param omega A float representing the angular frequency of the oscillator.
    @param debug A boolean to enable or disable debug messages.
    
    @return 
        - eigvals: A 1D numpy array containing the computed eigenvalues.
        - eigvecs: A 2D numpy array where each column is a normalized eigenvector.

    @note The function assumes the mass of the particle to be 1 and uses ħ=1 (natural units). Also the imaginary unit is considered 1.
    @warning The approximation order should be 2 or 3; any other value will raise an error.
    """

    #initializing the diagonals values depending on the approximation order
    if order==2:
        values_kinetic=[2,-1]
    elif  order==3:
        values_kinetic=[5/2, -4/3, 1/12]
    else: 
        error("Order not  valid: should be 2 (second) or 3 (third)")

    # Generating the hamiltonian matrix of the quantum harmonic oscillator
    # Discretized 1-D space domain
    a, b, N =x[0], x[-1], x.shape[0]
    deltaX = (b-a)/N 

    #Kinetic part of the hamiltonian, approximated to second order (See: "Finite difference method")
    prefactor = 1 / (2 * deltaX**2)
    K = prefactor * gen_diag_simmetric_mat(size=N, values=values_kinetic, sparse=sparse)

    #Potential part of the hamiltonian
    values_V = np.array([(1/2) * (omega**2) * x**2]) # main diagonal (not constant elements case)
    V = gen_diag_simmetric_mat(size=N, values = values_V, sparse=sparse)

    #The hamiltonian of the Quantum Harmonic Oscillator, approximated to second order via Finite difference method
    H = V + K

    # Diagonalize the Hamiltonian to get eigenvalues and eigenvectors  
    tic = time()
    eigvals, eigvecs = diagonalize_hermitian(mat=H, k=k)
    tac = time() 
    tictac=tac-tic
    checkpoint(f"Diagonalization have taken: {tictac} seconds", debug=debug)

    # Normalize eigenvectors
    for idx in range(eigvecs.shape[1]): #iterating on the columns of eivecs, i.e., on each eigenvector
        vec = eigvecs[:, idx]
        integral = np.sum((vec**2)*deltaX)
        eigvecs[:, idx] = (1 / np.sqrt(integral)) * vec

        # Check normalization (integral should be approximately 1)
        norm_check=np.sum((eigvecs[:, idx]**2) * deltaX)
        if not np.isclose(norm_check, 1.0, atol=1e-6):  
            error(f"Normalization check failed for eigenvector {idx}: norm = {norm_check}")
    
    return eigvals, eigvecs


    
def QHO_plot_eigenfunction(x, a, b, n, wave_func, wave_func_theory, main_fold):
    """
    @brief Plot the eigenfunction of the Quantum Harmonic Oscillator (QHO).

    This function visualizes the numerical and theoretical eigenfunctions for a specified energy level \(n\).
    It also marks the boundary points of the domain and saves the plot as an SVG file.

    @param x A 1D numpy array representing the discretized spatial domain (grid points).
    @param a A float representing the left boundary of the spatial domain.
    @param b A float representing the right boundary of the spatial domain.
    @param n An integer specifying the energy level of the eigenfunction to plot.
    @param wave_func A 1D numpy array containing the numerical eigenfunction values.
    @param wave_func_theory A 1D numpy array containing the theoretical eigenfunction values.
    @param main_fold A string representing the path to the main folder where the plot will be saved.

    @return Void. The function generates and saves a plot of the eigenfunction.

    @note Ensure the `main_fold` directory exists and has a subdirectory `plots` before calling this function.

    @warning 
    - The numerical and theoretical wave functions should be precomputed and aligned with the spatial domain `x`.
    """
    plt.figure(figsize=(10,10))

    plt.scatter(x, np.abs(wave_func)**2, color='darkblue', s=10, label='Numerical')
    plt.plot(x, np.abs(wave_func_theory)**2, color='gold', label='Theoretical')

    plt.axvline(a, color='purple', linewidth=3, linestyle='dashed', dashes=(5, 5)) 
    plt.axvline(b, color='purple', linewidth=3, linestyle='dashed', dashes=(5, 5))
    plt.ylabel(f'$|\Psi_{n}(x)|^2$ ')
    plt.xlabel('x')
    plt.title(f"QHO eigenfunction energy level n={n}")
    plt.legend()
    plt.grid('True')
    plt.savefig(main_fold+f'plots/QHO_eigenfunction_{n}'+'.svg', format='svg', bbox_inches='tight')
    plt.show()


                            
def QHO_numerical_VS_theory(a, b, N, k, sparse, main_fold, order=2, m=1, h_bar=1, omega=1, debug=False, plot=True):
    """
    @brief Compare numerical and theoretical solutions for the Quantum Harmonic Oscillator (QHO).

    This function computes the eigenvalues and eigenvectors of the QHO Hamiltonian numerically and compares them 
    with theoretical predictions. It calculates energy differences and overlaps between numerical and theoretical 
    eigenfunctions, and optionally generates plots for visualization.

    @param a A float representing the left boundary of the spatial domain.
    @param b A float representing the right boundary of the spatial domain.
    @param N An integer specifying the number of grid points in the spatial domain.
    @param k An integer specifying the number of eigenvalues and eigenvectors to compute and compare.
    @param sparse A boolean indicating whether to use sparse matrix representations.
    @param main_fold A string representing the path to the main folder where plots will be saved.
    @param order An integer specifying the finite difference approximation order (default: 2).
    @param m A float representing the mass of the particle (default: 1).
    @param h_bar A float representing the reduced Planck's constant (default: 1).
    @param omega A float representing the angular frequency of the oscillator (default: 1).
    @param debug A boolean to enable or disable debug messages (default: False).
    @param plot A boolean to enable or disable plotting of eigenfunctions (default: True).

    @return 
        - energy_errors: A 1D numpy array containing the energy differences between numerical and theoretical eigenvalues.
        - overlaps: A 1D numpy array containing overlaps between numerical and theoretical eigenfunctions.

    @note 
    - Ensure that the `main_fold` directory exists with a `plots` subdirectory if plotting is enabled.
    - The function assumes that \( \hbar = 1 \) and \( m = 1 \) by default unless specified otherwise.

    @warning 
    - Ensure the grid resolution \(N\) is sufficiently high for accurate numerical results.
    """
    x = np.linspace(a,b,num=N)
    deltaX = (b-a)/N

    #Computing the first k eigenvalues and eigenfunctions
    eigvals, eigvecs = QHO_diagonalization(x, k, sparse, order, omega, debug) 

    # Computing the theoretical expectations for the eigenvalues and eigenvectors
    theoretical_eigvals, theoretical_eigvecs = stationary_states(x, omega, m, h_bar, k)

    # Comparing theoretical with numerical results for each eigenvalue/eigenvector
    # computing the difference in the energies and the overlaps between numerical and theoretical eigenvectors
    energy_errors, overlaps = [],[]
    for n in range(k): #for each eigenvalue/eigenvector
        checkpoint('-----------------------------', debug=debug)
        checkpoint(f"Energy level {n}", debug=debug)

        #retrieving the eigenvalue and corresponding eigenvector
        energy, wave_func = eigvals[n], eigvecs[:, n]
        energy_theory, wave_func_theory = theoretical_eigvals[n], theoretical_eigvecs[:, n] #hbar and m = 1 by default
        
        # computing the energy difference between the numerical and theroetical estimates
        deltaE = np.abs(energy_theory-energy)
        checkpoint("Eigenvalue:", debug=debug)
        checkpoint(f"\tNumerical: {energy}", debug=debug)
        checkpoint(f"\tTheoretical: {energy_theory}", debug=debug)

        #Computing overlap of the two eigenvector estimates
        ovrlp = overlap(wave_func, wave_func_theory, deltaX) 
        checkpoint(f"Overlap:\t{ovrlp}", debug=debug)

        #storing results
        energy_errors.append(deltaE)
        overlaps.append(ovrlp)
    
        if plot:
            QHO_plot_eigenfunction(x, a, b, n, wave_func, wave_func_theory, main_fold)
    energy_errors, overlaps = np.array(energy_errors), np.array(overlaps)

    return energy_errors, overlaps


def QHO_errors_analysis(intervals_sizes, N_values, k, omega, sparse, main_fold, order=2, m=1, h_bar=1, plot_single_functions=False, debug=False):
    """
    @brief Perform an analysis of the energy errors and overlaps for the Quantum Harmonic Oscillator (QHO) eigenfunctions.

    This function computes the energy errors and overlaps for a range of grid sizes and interval sizes, comparing 
    the numerical and theoretical eigenvalues and eigenfunctions. It generates plots for energy errors and overlaps 
    and returns the results as pandas DataFrames.

    @param intervals_sizes A list or array of interval sizes to define the spatial domain (left and right boundaries).
    @param N_values A list or array of grid sizes (number of discretization points).
    @param k An integer specifying the number of eigenvalues and eigenvectors to compute and analyze.
    @param order An integer specifying the finite difference approximation order for numerical differentiation (e.g., 2 for second order).
    @param m A float representing the mass of the particle (default: 1).
    @param h_bar A float representing the reduced Planck's constant (default: 1).
    @param omega A float representing the angular frequency of the oscillator (default: 1).
    @param sparse A boolean indicating whether to use sparse matrix representations.
    @param main_fold A string representing the path to the main folder where plots will be saved.
    @param plot_single_functions A boolean to enable or disable plotting of individual eigenfunctions (default: False).
    @param debug A boolean to enable or disable debug messages (default: False).

    @return 
        - energy_errors_df: A pandas DataFrame containing the energy errors for each combination of interval size and grid size.
        - overlaps_df: A pandas DataFrame containing the overlaps for each combination of interval size and grid size.

    @details 
    - The function computes numerical eigenvalues and eigenvectors using the `QHO_numerical_VS_theory` function 
      for each combination of interval size and grid size.

    @note 
    - Ensure that the `main_fold` directory exists with a subdirectory `plots` before calling this function.
    - The function assumes the input values for mass \(m\), reduced Planck's constant \(\hbar\), and angular frequency \(\omega\) 
      are provided in natural units unless specified otherwise.

    @warning 
    - The grid resolution \(N\) should be large enough for accurate numerical results.
    """
    all_energy_errors, all_overlaps = [], []
    # Iterate over the combinations of interval sizes and grid sizes    
    for size, N in itertools.product(intervals_sizes, N_values):
        # Define the domain to be centered at 0
        a, b = -size/2, size/2 

        # Compute the energy errors and overlaps for the first k eigenvalues/eigenvectors, for the current combination.
        # and storing the results
        energy_errors, overlaps = QHO_numerical_VS_theory(a=a, b=b, N=N, k=k, sparse=sparse, order=order, m=m, h_bar=h_bar, omega=omega, debug=debug,  plot=plot_single_functions, main_fold=main_fold)
        all_energy_errors.append([a, b, N, energy_errors])
        all_overlaps.append([a, b, N, overlaps])
    # Convert the results into pandas DataFrames
    energy_errors_df = pd.DataFrame(all_energy_errors, columns=['a', 'b', 'N', 'energy_errors'])
    overlaps_df = pd.DataFrame(all_overlaps, columns=['a', 'b', 'N', 'overlaps'])

    # Plotting energy errors
    plt.figure(figsize=(10,10))
    #for each configuration of interval size and discretization, retrieve the energy errors and overlaps
    for i, e_errs in energy_errors_df.iterrows():
        # retrieve the configuration variables
        a, b, N = e_errs['a'], e_errs['b'], e_errs['N']

        #plotting error energies 
        plt.scatter(range(k), e_errs['energy_errors'], color=color_cycle[i], s=20, label=f'(a={a}, b={b}, N={N})')
        plt.plot(range(k), e_errs['energy_errors'], color=color_cycle[i])
    #setting plt parameters
    plt.ylabel('$|E_{n,theory} - E_{n,numerical}|$')
    plt.xlabel('Energy level $n$')
    plt.xticks(np.arange(stop=k, step=2))
    plt.title(f"Errors in the eigenvalues")
    plt.legend()
    plt.grid('True')
    plt.savefig(main_fold+f'plots/err_eigvals'+'.svg', format='svg', bbox_inches='tight')
    plt.show()

    # Plotting overlaps
    plt.figure(figsize=(10,10))
    #for each configuration of interval size and discretization, retrieve the energy errors and overlaps
    for j, ovrlps in overlaps_df.iterrows():
        # retrieve the configuration variables
        a, b, N = ovrlps['a'], ovrlps['b'], ovrlps['N']

        # plotting overlaps
        plt.scatter(range(k), ovrlps['overlaps'], color=color_cycle[j], s=20, label=f'(a={a}, b={b}, N={N})')
        plt.plot(range(k), ovrlps['overlaps'], color=color_cycle[j])
    #setting plt parameters
    overlap_formula = r'$O = \left| \int \psi_{\text{num}}^*(x) \, \psi_{\text{theo}}(x) \, dx \right|$'
    plt.ylabel(overlap_formula)
    plt.xlabel('Energy level $n$')
    plt.xticks(np.arange(stop=k, step=2))
    plt.title(f"Overlaps")
    plt.legend()
    plt.grid('True')
    plt.savefig(main_fold+f'plots/overlaps'+'.svg', format='svg', bbox_inches='tight')
    plt.show()

    return energy_errors_df, overlaps_df