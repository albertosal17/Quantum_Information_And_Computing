import numpy as np
import matplotlib.pyplot as plt
from time import time
from scipy.optimize import curve_fit

from N_body_wfc import generate_Nbody_wfc, Nbody_separable_wfc
import debugger_module as dbg 

color_cycle = plt.cm.Set1.colors  # Setting a predefined colormap for the plots
plt.rcParams.update({'font.size': 24}) #setting font size for the plots

debug=True

D=2
N=np.arange(2, 40, dtype=int)
max_N_general = 25
num_measures = 7  # Number of timing measurements per N

avg_timings = np.zeros((N.shape[0], 2))  # Average timings
std_timings = np.zeros((N.shape[0], 2))  # Standard deviations
ii = 0
for nn in N:
    dbg.checkpoint(f"N={nn}", debug=debug)
    general_times = []
    separable_times = []

    for _ in range(num_measures):

        if nn<max_N_general+1:
            tic = time()
            general = generate_Nbody_wfc(D,nn)
            general_times.append( time() - tic)
        tic = time()
        separable = Nbody_separable_wfc(D,nn, compute_total_state=False)
        separable_times.append( time() - tic )

    # Compute averages and standard deviations
    if nn < max_N_general + 1:
        avg_timings[ii, 0] = np.mean(general_times)
        std_timings[ii, 0] = np.std(general_times)
    avg_timings[ii, 1] = np.mean(separable_times)
    std_timings[ii, 1] = np.std(separable_times)


    ii+=1

N_general = N[:(max_N_general-1)]
timings_general = avg_timings[:(max_N_general-1),0]
timings_separable= avg_timings[:,1]

# Define the fitting functions
def general_fit(N, a, b):
    return a * N + b  # Linear in log-scale

def separable_fit(N, a, b):
    return a * np.log(N) + b  # Logarithmic growth

# Perform fits
popt_general, _ = curve_fit(general_fit, N_general, np.log(timings_general))
popt_separable, _ = curve_fit(separable_fit, N, np.log(timings_separable))

# Generate fitted lines for plotting
N_fit_general = np.linspace(N_general[0], N_general[-1], 100)
N_fit_separable = np.linspace(N[0], N[-1], 100)
fit_general = np.exp(general_fit(N_fit_general, *popt_general))  # Convert back from log scale
fit_separable = np.exp(separable_fit(N_fit_separable, *popt_separable))

plt.figure(figsize=(12,8))
plt.errorbar(N_general, timings_general, yerr=std_timings[:(max_N_general - 1), 0], fmt='o',color='darkred', label='general_wfc')
plt.errorbar(N, timings_separable, yerr=std_timings[:, 1],fmt='o', color='darkblue', label='separable_wfc')

# Plot the fitted lines
plt.plot(N_fit_general, fit_general, label=f'y={np.round(popt_general[0],2)}*N {np.round(popt_general[1],2)}', linewidth=1, linestyle='dashed', color='darkred')
plt.plot(N_fit_separable, fit_separable, label=f'y={np.round(popt_separable[0],2)}*log(N) {np.round(popt_separable[1],2)}', linewidth=1, linestyle='dashed',color='darkblue')


plt.xlabel('Number of subsystems N')
plt.ylabel('CPU_time (s)')
plt.title(f'CPU_time vs. number of subsystems (D={D})')
plt.grid('True')
plt.legend()
plt.yscale('log')
plt.savefig(f'/home/albertos/quantumInfo/ex6/plots/benchmarking_total_wfc_D{D}.svg', format='svg', bbox_inches='tight')

plt.show()