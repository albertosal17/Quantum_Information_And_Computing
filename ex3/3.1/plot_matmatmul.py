import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
from datetime import datetime
from scipy.optimize import curve_fit
from debugger_module import checkpoint, error, warning

plt.rcParams.update({'font.size': 18}) #setting font size for the plots

debug = False # Choose wheter to activate debugging mode


def plot_method(df_method, method_index, color_index, superimpose):
    """
    @brief Plots execution time vs. matrix size for a given matrix-matrix multiplication method, with error bars and a power-law fit.

    This function takes a DataFrame containing averages and standard deviations for the execution time for different matrix sizes,
    plot this values appliying error bars, and fits the data with a power-law model (cubic law) to estimate the 
    relationship between matrix size and execution time. The plot is customized with color and 
    labels based on the method provided.

    @param df_method [pandas.DataFrame] DataFrame containing execution time data.
                       Expected columns: 
                       - Column 0: Matrix size (N)
                       - Column 1: Average execution time (avg_t)
                       - Column 2: Standard deviation of execution time (std_t)

    @param method_index [str] Identifier for the method being plotted. Options:
                                'M1' for "row-by-col": standard matrix multiplication algorithm that makes
                                scalar product between the leftmost matrix rows and right-most matrix columns, 
                                'M2' for "col-by-col": optimize method for fortran computation that firstly compute
                                the transpose of the left-most matrix and then computes the multiplication as a 
                                scalar product between the columns of the original matrices,
                                'M3' for "built-in": using the built-in function of Fortran for the matrix multiplication

    @param color_index [int] Index used to select the color from the predefined colormap (Paired).
                             Ensures different methods are distinguishable on the plot.
    @param superimpose [bool] Flag to be set to True if one want a final plot with the curves for each method superimposed
                                otherwise it will produce an unique plot for each method

    @throws ValueError if the DataFrame does not have exactly three columns.
    """

    if not superimpose:
        plt.figure(figsize=(8,8))

    #assigning a more meaningful name to the method
    if method_index == 'M1':
        method_label =  "'row-by-col'"
    elif method_index == 'M2':
        method_label =  "'col-by-col'"
    elif method_index == 'M3':
        method_label =  "'built-in'"
    else:
        print("WARNING: Unknown method")
    
    #plt.figure(figsize=(10, 6))
    color_cycle = plt.cm.Paired.colors  # Setting a predefined colormap for the plots

    # Retrieving arrays from the dataframe: sizes, averages and std deviations fro the execution times
    if df_method.shape[1] != 3: #checking if the dataframe has appropriate shape: should have three columns [N,avg_t,std_t]
        error(f"Invalid dataframe shape: {df_method.shape[1]} columns (should be 3)")
    N_values=df_method.iloc[:, 0].values
    t_values=df_method.iloc[:, 1].values
    err_t = df_method.iloc[:, 2].values

    # Plotting data points matrix_size VS. average_execution_time
    plt.errorbar(N_values, t_values, yerr=err_t, fmt='o', label='method '+method_label, capsize=3, lw=1, markersize=6, color=color_cycle[color_index])


    # Fitting data points with a power law: we expect from theory a cubic law for the number of operations (and so the execution time) with respect to the size of the matrix
    def power_law(x, a, b):
        return a * np.power(x, b)
    
    popt, _ = curve_fit(power_law, N_values, t_values)

    x_log = np.logspace(np.log10(min(N_values)), np.log10(max(N_values)), 100)

    plt.plot(x_log, power_law(x_log, *popt), label="({0:.3e}*x**{1:.3e})".format(*popt), linewidth=1, linestyle='dashed', color=color_cycle[color_index+1])
    
    checkpoint("Exponential Fit: y = (a*(x**b)) method " + method_label, debug=debug)
    checkpoint("\ta = {0}\n\tb = {1}".format(*popt), debug=debug) 

    # This is the case where you get different plots for each method
    if not superimpose:
        plt.xlabel('matrix size N')
        plt.ylabel('CPU time (s)')
        plt.title('Execution time scaling using method ' + method_label)
        plt.legend()
        plt.grid(True)
        plt.yscale('log')
        plt.xscale('log')
        now = datetime.now().strftime("%Y-%m-%d_%H-%M")  # Current date and time
        #plt.savefig(main_folder+'method_'+method_index+'_'+now+'.jpeg')  # Show only for individual plots
        plt.savefig(main_folder+'method_'+method_index+'_'+now+'.svg', format='svg', bbox_inches='tight')
        plt.show()

    return


# specifying the folder where data are contained and where to store plots
main_folder = './data/'

# storing the paths of the files present in th emain folder
data_files = [os.path.join(main_folder, file) for file in os.listdir(main_folder) if file.endswith('.csv') and os.path.isfile(os.path.join(main_folder, file))]
checkpoint("Files present in the folder: ", debug=debug, data_files=data_files)

# setting parameters
superimpose=True # flag to be turned on wheter you want to plot the various curves superposed in a single figure or instead separate figures for each plot
color_index=0 # index used for accessing the color map and selecting the color to be used for the curves in the plots
if superimpose:
    plt.figure(figsize=(12,8))
# reading all the files inside the folder and plotting the relative matrix_size vs. execution_time curve
for data_file in data_files:

    if not 'execution_times_M' in data_file:
        error("Cannot find any data file with appropriate name")

    #retrieving the method index (e.g. 'M1', 'M2', ..) from the file name
    method_index = data_file[data_file.find('execution_times_') + len('execution_times_'):data_file.find('.csv')]

    # Loading the data inside the file in a dataframe
    df_raw = pd.read_csv(data_file, sep='\s+') #sep='\s+' is the new way for saying delim_whitespace=True (for newer pandas distributions)

    #checking if the dataframe has appropriate shape: should have two columns [N,t]
    if df_raw.shape[1] != 2: 
        error(f"Invalid dataframe shape: {df_raw.shape[1]} columns (should be 2)")
    df_raw.columns = ['N', 't'] #assigning name to the columns

    #computing average times and standard deviations for each N 
    df_raw = df_raw.sort_values(by='N') 
    gb = df_raw.groupby('N')['t']
    avgs = gb.mean()
    stds = gb.std()
    warning("If an N configurations as been executed only once, the corresponding standard deviation will be NaN. Future improvements of the ccode may be necessary.",debug=debug)

    # creating a new dataframe with entries [N, average_time, standar_deviation_time]
    final_df = pd.concat([avgs, stds], axis=1)
    final_df.columns = ['avg_'+method_index, 'std_'+method_index]
    final_df = final_df.reset_index()

    # Plotting the curve execution time vs. matrix size
    plot_method(final_df, method_index, color_index=color_index, superimpose=superimpose)

    color_index+=2

# This is the case where you get an unique plot superimposing the results
if superimpose:
    plt.xlabel('matrix size N')
    plt.ylabel('CPU time (s)')
    plt.title('Execution time scaling comparison')
    plt.legend()
    plt.grid(True)
    plt.yscale('log')
    plt.xscale('log')
    now = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")  # Current date and time
    #plt.savefig(main_folder+'methods_comparison_'+now+'.jpeg')  # Show only for individual plots
    plt.savefig(main_folder+'methods_comparison_' + now +'.svg', format='svg', bbox_inches='tight')

    plt.show()


