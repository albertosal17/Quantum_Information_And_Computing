import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from scipy.stats import poisson
from scipy.interpolate import interp1d
import os

plt.rcParams.update({'font.size': 16}) #setting font size for the plots


def gen_hermitian(N, mean_gaussian=0, var_gaussian=1):
    """
    @brief Generates a real random Hermitian matrix of size NxN.
    
    @param N The size of the Hermitian matrix (number of rows and columns).
    @param mean_gaussian The mean of the Gaussian distribution used for generating random entries (default is 0).
    @param var_gaussian The variance of the Gaussian distribution used for generating random entries (default is 1).
    
    @return A real Hermitian matrix of size NxN with random entries.
    
    The function constructs an NxN matrix by filling the upper triangular part (including the diagonal) with random
    values sampled from a Gaussian distribution defined by the `mean_gaussian` and `var_gaussian` parameters. The 
    lower triangular part is filled to ensure the matrix is Hermitian (i.e., \( M[i, j] = M[j, i] \)).
    """
    # Creating an empty matrix
    M = np.empty((N, N), dtype=float)

    # Generating random entries for the upper triangular+diagonal
    upper_triangle_size = int(N * (N + 1) / 2)  
    upper_triangle_flat = np.random.normal(loc=mean_gaussian, scale=np.sqrt(var_gaussian), size=upper_triangle_size) 
   
    # Filling the matrix upper triangular part + diagonal
    M[np.triu_indices(N)] = upper_triangle_flat #np.triu_indeces(N) return the indeces of the upper triangle plus diagonal of an N by N matrix
    
    # Filling the lower triangular part exploiting hermitianicity
    lower_triangle = M.T - np.diag(M.diagonal()) #np.diag() generates a diagonal matrix provided the 1d diagonal
    M = M + lower_triangle

    return M

def gen_random_diag(N, mean_gaussian=0, var_gaussian=1):
    """
    @brief Generates a random diagonal matrix of size NxN with Gaussian-distributed diagonal entries.
    
    @param N The size of the matrix (number of rows and columns).
    @param mean_gaussian The mean of the Gaussian distribution used for generating diagonal entries (default is 0).
    @param var_gaussian The variance of the Gaussian distribution used for generating diagonal entries (default is 1).
    
    @return A diagonal matrix of size NxN with entries drawn from a Gaussian distribution.
    
    The function creates an NxN matrix with only diagonal elements populated by random values sampled from a 
    Gaussian distribution defined by `mean_gaussian` and `var_gaussian`. The off-diagonal elements are zero.
    """
    diag = np.random.normal(loc=mean_gaussian, scale=np.sqrt(var_gaussian), size=N)
    mat_rand = np.diag(diag)

    return mat_rand

def compute_eigenspaces(matrix):
    """
    @brief Computes the normalized spacings between the unique eigenvalues of a matrix.
    
    @param matrix A square matrix (NxN) for which the eigenvalues and their spacings are computed.
    
    @return An array containing the normalized spacings between the unique eigenvalues of the input matrix.
    
    The function calculates the eigenvalues of the input matrix and filters them to retain only unique values 
    (i.e., removes eigenvalues with multiplicity greater than 1). The unique eigenvalues are sorted in ascending order, 
    and the spacings between consecutive eigenvalues are computed. The average spacing is used to normalize these spacings.

    Steps performed by the function:
    - Computes the eigenvalues of the matrix.
    - Filters out duplicate eigenvalues and sorts them.
    - Calculates the spacings between consecutive eigenvalues.
    - Normalizes the spacings by dividing them by the mean spacing.
    """

    #computing the eigenvalues
    eigs, _ = np.linalg.eig(matrix)

    #filtering the eignevalues to remove the ones with multiplicity > 1 and sorting in ascending order
    eigs = np.unique(eigs)
    eigs = np.sort(eigs)

    #computing the spacings between the eigenvalues of the matrix
    eigs_shifted = np.roll(eigs,shift=-1) #shift the array of 1 position to the left
    spacings= np.abs(eigs_shifted-eigs) #spacing is the absolute difference between the values
    spacings= spacings[:-1] #removing the unuseful spacing between the first and last element

    #computing the average spacing
    avg_space = spacings.mean()

    #Compute the normalized spacing: by definition they are normalized to the mean value
    normalized_spacings = spacings/avg_space

    return normalized_spacings

def wigner_surmise(s):
    return (32 / np.pi**2) * s**2 * np.exp(-4 * s**2 / np.pi)

def poisson_distr(x, lam):
    return poisson.pmf(x, lam)

def plot_histogram_spacings(spacings, name):
    """
    @brief Plots a histogram of normalized eigenvalue spacings and fits it to an appropriate distribution.
    
    @param spacings A NumPy array containing the normalized spacings between eigenvalues.
    @param name A string indicating the type of matrix ('Hermitian' or 'Random diagonal') to choose the distribution for fitting.
    
    @return None. The function displays the plot and saves it as an image file.
    
    The function generates a histogram of the normalized eigenvalue spacings with error bars and fits it to a specific 
    distribution based on the type of matrix:
    - For 'Hermitian' matrices, it fits a Wigner Surmise distribution.
    - For 'Random diagonal' matrices, it fits a Poisson distribution, interpolating to create a smooth curve.

    The function performs the following:
    - Computes and plots a histogram of the input spacings.
    - Calculates Poissonian error bars for the histogram.
    - Fits the histogram to a Wigner Surmise or Poisson distribution based on the `name` parameter.
    - Saves the plot as an image file named `./<name>.jpeg`.
    """
    plt.figure(figsize=(8,8))
    
    # Creating histogram and get densities and bin edges
    densities, bin_edges, _ = plt.hist(spacings, fc="yellow", ec='grey', bins=14, density=True)

    # Calculating the bin error bars and plotting them
    bin_centers = 0.5 * (bin_edges[1:] + bin_edges[:-1])
    bin_width = bin_edges[1] - bin_edges[0]
    N = len(spacings) #number of data points
    frequencies = densities * N * bin_width #need to go to frequencies and the go back to densities in order to exploit Poissonian errors as frequencies**0.5
    errors = np.sqrt(frequencies) / (N * bin_width)
    plt.errorbar(bin_centers, densities, yerr=errors, fmt='.', color='black', capsize=3)

    # Different fit distribution function for different random matrix spacings' distribution
    #       Hemitian random matrix spacings --> Wigner Surmise distribution
    #       Random diagonal matrix spacings --> Poisson distribution
    if name=='Hermitian':
        # Plotting the Wigner Surmise
        s_values = np.linspace(min(spacings), max(spacings), 100)     
        plt.plot(s_values, wigner_surmise(s_values), 'b-', label='Wigner Surmise distribution')

    if name=='Random diagonal':
        #Plotting an interpolating continuous function based on the (discrete) poisson distribution

        # Retrieving poisson parameter by fitting data
        popt, _ = curve_fit(poisson_distr, bin_centers, densities)
        mu = popt[0]     

        # Computing the actual poisson distribution for our spacings
        s_values_discrete = np.arange(0, 11, 1) # values for the discrete PMF
        pmf_values = poisson_distr(s_values_discrete, mu)

        # Interpolate the poisson discrete distribution to create a smooth curve
        s_values_continuous = np.linspace(0, 10, 500)
        interpolator = interp1d(s_values_discrete, pmf_values, kind='cubic')
        pmf_smooth = interpolator(s_values_continuous)

        # plotting the results
        plt.plot(s_values_continuous, pmf_smooth, 'b-', label=f'Poisson distribution $\lambda$={mu}')
    

    plt.xlabel('Normalized eigen-spacing')
    plt.ylabel('Probability Density')
    plt.title('Spacing distribution - '+ name)
    plt.legend()
    #plt.savefig('./'+name+'.jpeg') 
    plt.savefig('./'+name+'.svg', format='svg', bbox_inches='tight')


    plt.show()

    return 


#Setting parameters
N=100
N_meas=1000

## Making repeated measures for different random matrices spacings, with fixed matrix size. All the results are then stored in an unique array.
#generating NaN arrays to be filled with measures
eigenspaces_samples_herm = np.full((N_meas, N-1), np.nan) #the max number of spacing is N-1 as N is the max number of different eigenvalues for an N by N matrix
eigenspaces_samples_rand = np.full((N_meas, N-1), np.nan) 
for i in range(N_meas):
    #Generating the matrices
    M_herm = gen_hermitian(N)
    M_rand = gen_random_diag(N)

    #Computing the spacings
    norm_spacings_herm = compute_eigenspaces(M_herm)
    norm_spacings_rand = compute_eigenspaces(M_rand)

    # Filling the array containing the measures
    eigenspaces_samples_herm[i, :len(norm_spacings_herm)] = norm_spacings_herm
    eigenspaces_samples_rand[i, :len(norm_spacings_rand)] = norm_spacings_rand
#flattening and removing eventualy NaN entries
eigenspaces_samples_herm = eigenspaces_samples_herm.flatten()
eigenspaces_samples_herm = eigenspaces_samples_herm[~np.isnan(eigenspaces_samples_herm)] 
eigenspaces_samples_rand = eigenspaces_samples_rand.flatten()
eigenspaces_samples_rand = eigenspaces_samples_rand[~np.isnan(eigenspaces_samples_rand)] 


# Plotting the distributions of spacings
plot_histogram_spacings(eigenspaces_samples_herm, name='Hermitian')
plot_histogram_spacings(eigenspaces_samples_rand, name='Random diagonal')


