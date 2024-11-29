import subprocess
import numpy as np

#the path to the executable
path_exe = "./matmatmul.exe"

#defining the range for the size of the matrix [N_min,N_max]
N_range = [100, 2300]  
N = np.linspace(start=N_range[0], stop=N_range[1], num=7).astype(int)

# Initializingthe input variables the fortran program requires
debug = ".FALSE." 
verbosity = 0 #integer in {0,1,2} 
rep_meas = 2 

#taking the measures
#the data will bee stored in .txt files stored inside './data/' folder
for n in N:
    # giving as input from the terminal the choosen parameters
    input_from_terminal = str(n)+"\n"+debug+"\n"+str(verbosity)+"\n"+str(rep_meas)

    #executing bash command to run the fortran program with the specified input
    subprocess.run(path_exe, input=input_from_terminal, text=True) 


