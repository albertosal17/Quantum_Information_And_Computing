import numpy as np
import matplotlib.pyplot as plt

from quantum_harmonic_oscillator_methods import QHO_numerical_VS_theory, QHO_errors_analysis

plt.rcParams.update({'font.size': 24}) #setting font size for the plots


### SINGLE CONFIGURATION OF INTERVAL SIZE AND DISCRETIZATION STEP ANALYSIS
# Initializing parameters 
a,b = -10, 10
N = 1000 
m, h_bar, omega = 1, 1, 1
k = 10  #number of eigenvalues/eigenvectors to be analyzed
order=2 #order of approcimation in the "finite difference method", used for computing the hamiltonian matrix
# Initializing boolean flags
sparse = True
plot = False
debug = False
main_fold ='/home/albertos/quantumInfo/ex4/'

_, _ = QHO_numerical_VS_theory(a=a, b=b, N=N, k=k, sparse=sparse, order=order, m=m, h_bar=h_bar, omega=omega, debug=debug, plot=plot, main_fold=main_fold)

###---------------------------------------------------------------------------------------
### MULTIPLE CONFIGURATIONS OF INTERVAL SIZES AND DISCRETIZATION STEPS (ERRORS) ANALYSIS

#Re-inizializing some of the old parameters
k=20
debug = False
# Inizializing new parameters
intervals_sizes = np.array([20,40,60])
N_values = np.array([500,1000,1500])
plot_single = False

energy_errors_df, overlaps_df = QHO_errors_analysis(intervals_sizes=intervals_sizes, N_values=N_values, k=k, order=order, m=m, h_bar=h_bar, omega=omega, sparse=sparse, main_fold=main_fold, plot_single_functions=plot_single, debug=debug)


 