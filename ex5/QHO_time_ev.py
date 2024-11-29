import numpy as np
from math import factorial
import matplotlib.pyplot as plt
from matplotlib import animation
from numpy.linalg import norm
import math

import debugger_module as dbg


def hermite(x, n):
    """
    hermite polynomials of order n
    defined on the real space x

    Parameters
    ----------
    x : np.ndarray
        Real space
    n : int
        order of the polynomial

    Returns
    -------
    np.ndarray
        Hermite polynomial of order n
    """
    herm_coeffs = np.zeros(n+1)
    herm_coeffs[n] = 1
    return np.polynomial.hermite.hermval(x, herm_coeffs)

def stationary_state(x,n,m):
    """
    Returns the stationary state of order
    n of the quantum harmonic oscillator

    Parameters
    ----------
    x : np.ndarray
        Real space
    n : int
        order of the stationary state. 0 is ground state.

    Returns
    -------
    np.ndarray
        Stationary state of order n
    """
    prefactor = 1./np.sqrt(2.**n * factorial(n)) * (m/(np.pi))**(0.25)
    psi = prefactor * np.exp(- m*x**2 / 2) * hermite(np.sqrt(m)*x,n)
    return psi

class Param:
    """
    Container for holding all simulation parameters

    Parameters
    ----------
    xmax : float
        The real space is between [-xmax, xmax]
    num_x : int
        Number of intervals in [-xmax, xmax]
    dt : float
        Time discretization
    T : int
        period
    im_time : bool, optional
        If True, use imaginary time evolution.
        Default to False.
    """
    def __init__(self,
                 xmax: float,
                 num_x: int,
                 T: int,
                 m:float,
                 timesteps: int,
                 im_time: bool = False) -> None:
        
        self.im_time = im_time   
        self.m=m

        #space grid
        self.xmax = xmax
        self.num_x = num_x
        self.dx = 2 * xmax / num_x
        self.x = np.arange(-xmax + xmax / num_x, xmax, self.dx)

        #time grid
        self.T = T
        self.timesteps = timesteps
        self.dt = T/timesteps
        self.t = np.arange(0, T, self.dt)  #discretized times

        #momentum grid
        self.dk = np.pi / xmax
        self.k = np.concatenate((np.arange(0, num_x / 2), np.arange(-num_x / 2, 0))) * self.dk


class Operators:
    """Container for holding operators and wavefunction coefficients."""
    def __init__(self, res: int) -> None: #res is the number of points in the discretization

        #self.V = np.empty(res, dtype=complex) #potential operator
        self.R = np.empty(res, dtype=complex) ## time propagator component of the B-K-H decomposition, potential part (R stands for 'real' as here the variable is the space)
        self.K = np.empty(res, dtype=complex) ## time propagator component of the B-K-H decomposition, kinetic part (k stands represent the fact that here the variable is the momentum)
        self.wfc = np.empty(res, dtype=complex) # wave function


#difining a function for initializing the wavefunct and potential of the QHO 
def init(par: Param, n:int , voffset: float=0.0, wfcoffset: float=0.0) -> Operators:
    """
    Initialize the wavefunction coefficients and the potential.

    Parameters
    ----------
    par: Param
        Class containing the parameters of the simulation
    voffset : float, optional
        Offset of the quadratic potential in real space
        Default to 0.
    wfcoffset: float, optional
        Offset of the wavefunction in real space. Default to 0.
    n : int, optional
        Order of the hermite polynomial (i.e. level of eigenstate)

    Returns
    -------
    Operators
        Filled operators
    """
    opr = Operators(len(par.x))

    # Quadratic potential
    opr.V = (0.5/par.m)* (par.x - voffset) ** 2 #omega=1
    opr.wfc = stationary_state(par.x-wfcoffset, n, par.m).astype(complex)
    opr.wfc /= np.sqrt(np.sum(np.abs(opr.wfc)**2)*par.dx) #normalization

    # coeffincient of the time propagator exponential operator
    # if we want imaginary time we set it to the complex unit so that the time variable becomes i*t ("tau")
    coeff = 1 if par.im_time else 1j  

    # CREATE THE EXPONENTIAL PART OF THE HAMILTONIAN (we exploit the Hausdorff expansion of the matrix exponential e^{A+B}\sim e^{A}e^{B}+ o(dt^{2}))
    ###
    # Vedi la formula decomposizione Suzuki-trotter dalle slide di Siloi
    ###
    # Operator in momentum space 
    opr.K = np.exp(-(0.5/m) * (par.k ** 2) * par.dt * coeff) #(p^2/2m) * (deltaT/2) con m=1 
    # Operator in real space
    opr.R = np.exp(-0.5 * opr.V * par.dt * coeff)

    return opr

# Plot the average position of the particle as a function of 𝑡.
def average_position(par: Param, opr: Operators) -> float:
    """Calculate the average position <Psi|H|Psi>."""
    # Creating real, momentum, and conjugate wavefunctions.
    wfc_r = opr.wfc
    wfc_c = np.conj(wfc_r)

    # Finding the momentum and real-space energy terms
    x_avg = wfc_c * par.x * wfc_r

    # Integrating over all space
    x_avg_final = sum(x_avg).real

    return x_avg_final * par.dx

def plot_frame(x, n_frame, res, n,m):
    plt.figure(figsize=(8,8))
    plt.plot(x, res[0,n_frame,:], lw=3, alpha=.5, linestyle='--', label='wfc')
    plt.plot(x, np.abs(stationary_state(x,n=n, m=m))**2, lw=3, alpha=.5, linestyle='-', label='stationary_state(n=0)')
    plt.plot(x,  res[2,n_frame,:], label='potential')
    plt.ylim([0,1])
    plt.legend()
    plt.title(f'frame {n_frame}')
    plt.show()

def split_op(par: Param, opr: Operators, q_0t_values: np.ndarray, plot_frames: bool, n:int, voffset: float=0.0, debug: bool=False) -> None:
    """
    Split operator method for time evolution.
    Works in place

    Parameters
    ----------
    par : Param
        Parameters of the simulation
    opr : Operators
        Operators of the simulation

    Returns
    -------
    None
        Acts in place
    """

    # Initialize res to store real and momentum space densities separately AND the potential
    #
    # Per ciascuna delle due componenti di res ci sarà una matrice con un numero di righe pari
    # al numero di timesteps e un numero di colonne pari al numero di punti in cui è discretizzato
    # il dominio delle posizioni
    res, avg_position_values = np.zeros((3, par.timesteps+1, par.num_x)), np.zeros((par.timesteps+1))
    #dbg.checkpoint(f"initial norm: {np.sum(np.abs(opr.wfc)**2)*par.dx}", debug=debug)

    # Storing the initial configurations of wavefunctions and potential
    res[0,0,:] = np.abs(opr.wfc)**2
    res[1,0,:] =  np.abs(np.fft.fft(opr.wfc))**2
    res[2,0,:] = opr.V
    avg_position_values[0] = average_position(par, opr)
    if plot_frames:
        plot_frame(par.x, n_frame=0, res=res, n=n,m=par.m)


    #### DACANCELLARE
    stopping_index = q_0t_values.shape[0]//2
    q_0t_values[stopping_index+1:] = q_0t_values[stopping_index]
    ###


    #### CHECK CHE q_0t_values ABBIA DIMENSIONE par.timesteps
    if q_0t_values.shape[0] != par.timesteps:
        dbg.error(f"Invalid shape for q_0t_values: should have dimension {par.timesteps} while it has dimension {q_0t_values.shape[0]}")
    else:
        for i in range(par.timesteps): #for each timestep
            #updating the potential and consequently the time propagator in the space of positions
            opr.V = (0.5/par.m) * (par.x - voffset - q_0t_values[i]) ** 2 #omega=1

            coeff = 1 if par.im_time else 1j  
            opr.R = np.exp(-0.5 * opr.V * par.dt * coeff)

            # Half-step in real space
            opr.wfc *= opr.R #evoluzione nello spazio delle posizioni

            # FFT to momentum space
            opr.wfc = np.fft.fft(opr.wfc) #passaggio allo spazio dei momenti

            # Full step in momentum space
            opr.wfc *= opr.K #evoluzione nello spazio dei momenti

            # iFFT back
            opr.wfc = np.fft.ifft(opr.wfc) #ritorno allo spazio delle posizioni

            # Final half-step in real space
            opr.wfc *= opr.R #evoluzione finale nello spazio delle posizioni
            # Ora hai trasformato i coefficienti della wavefunction a seguito di un evoluzione di 1 timestep
            
            # Density for plotting and potential
            density = np.abs(opr.wfc) ** 2
            density_momentum = np.abs(np.fft.fft(opr.wfc))**2

            #  Renormalizing the wavefunction opr.wfc
            renorm_factor = np.sum( density * par.dx )
            opr.wfc /= np.sqrt(renorm_factor)

            norm_psi = np.sum( (np.abs(opr.wfc) ** 2) * par.dx )
            if not (math.isclose(norm_psi, 1.0, rel_tol=1e-6)):
                dbg.checkpoint(f"Check norm: {norm_psi}", debug=True)

            # Storing the wavefunctions and the potential
            res[0, i, :] = np.real(density)  # Real space
            res[1, i, :] = np.real(density_momentum)  # Momentum space
            res[2, i, :] = np.real(opr.V)  # potential
            avg_position_values[i] = average_position(par, opr)

    return res, avg_position_values


def calculate_energy(par: Param, opr: Operators) -> float:
    """Calculate the energy <Psi|H|Psi>."""
    # Creating real, momentum, and conjugate wavefunctions.
    wfc_r = opr.wfc
    wfc_k = np.fft.fft(wfc_r)
    wfc_c = np.conj(wfc_r)

    # Finding the momentum and real-space energy terms
    energy_k = 0.5 * wfc_c * np.fft.ifft((par.k ** 2) * wfc_k) ##CONTROLLA SE CI VA LA MASSA
    energy_r = wfc_c * opr.V * wfc_r ##CONTROLLA SE CI VA LA MASSA

    # Integrating over all space
    energy_final = sum(energy_k + energy_r).real

    return energy_final * par.dx


############################################################################################
##
## Given |Ψ0⟩ = |𝑛 = 0⟩ (ground state of the Harmonic oscillator), compute |Ψ(𝑡)⟩ for different values of 𝑇. 
## Plot the square norm of |Ψ(𝑡)⟩ as a function of 𝑞 at different times
## and the average position of the particle as a function of 𝑡.
##


#initializing variables
nx = 1000 #numero di punti in cui è discretizzato lo spazio delle posizioni
xmax = 4 # limite destro intervallo spazio delle posizioni
nt = 30000 # timesteps = T / dt 
tmax = 10000 # T
m=1 ### CONTROLLA BENE CHE LA MASSA SIA CORRETTAMENTE APPLICATA IN GIRO SE VUOI CAMBIARNE VALORE DA 1 
 
im_time=False #boolean flag to set imaginary-time evolution
debug=True
plot_frames = False

# wavefunction
n=0 # livello energetico dell'oscillatore armonico da simulare. 0 è il ground state
wfcoffset=0 #offset della funzione d'onda rispetto l'origine

voffset=0

dbg.checkpoint("Inizializad variables", debug=debug)


# parametri per discretizzazione
params = Param(xmax=xmax, num_x=nx, timesteps=nt, T=tmax, im_time=im_time, m=m) 
dbg.checkpoint(f"Time-step size: dt={round(params.dt,4)}", debug=debug)
q_0t_values = params.t / tmax


# inizializzazione funzione d'onda (al ground state) e operatori evoluzione temporale
ops = init(params, n, voffset, wfcoffset) #operatori e funzione d'onda QHO
#plotting the inizialized state
""" plt.figure(figsize=(8,8))
plt.plot(params.x, np.abs(ops.wfc)**2, lw=3, alpha=.5, linestyle='--', label='wfc')
plt.plot(params.x, np.abs(stationary_state(params.x,n=n))**2, lw=3, alpha=.5, linestyle='-', label='stationary_state(n=0)')
plt.plot(params.x,  ops.V, label='potential')
plt.ylim([0,1])
plt.legend()
plt.title('initialized state')
plt.show() """

# Time evolution
res, avg_position_values = split_op(par=params, opr=ops, q_0t_values=q_0t_values, debug=debug, plot_frames=plot_frames, n=n)
dbg.checkpoint("Evolution computed", debug=debug)
if plot_frames:
    plot_frame(params.x, n_frame=1, res=res, n=n,m=m) #intermediate evolution time
    plot_frame(params.x, n_frame=nt//2, res=res, n=n,m=m) #intermediate evolution time
    plot_frame(params.x, n_frame=2*(nt//3), res=res, n=n, m=m) #past-intermediate evolution time


# Plotting the time evolution 
step_interval = max(params.timesteps // 200, 1)  # Ensure at least 200 frames, adjust as needed (20000//500=40)
selected_frames = range(0, params.timesteps+1, step_interval)

fig, ax = plt.subplots()
ax.set_xlim(-params.xmax, params.xmax)
ax.set_ylim(0, 1)
ax.set_xlabel("Position (x)")
ax.set_ylabel("Probability Density |ψ(x)|^2")

line_V, = ax.plot([], [], lw=2, label="Potential V(x)", color='gray', linestyle='--')
line, = ax.plot([], [], lw=2, label="Wave Function |ψ(x)|^2", color='blue')
vertical_line, = ax.plot([], [], lw=2, color='red', label='Average position')
ax.legend()

# Initialization function for the animation
def init_lines():
    line.set_data([], [])
    line_V.set_data([], [])
    vertical_line.set_data([], [])

    return line, line_V, vertical_line

# Animation function that updates the plot at each frame
def animate(i):
    x = params.x
    y = res[0, i, :]
    y_V = res[2, i,:]

    #vertical line for the average position
    x_vertical = [avg_position_values[i]]  # Vertical line x-position at timestep i
    y_vertical = [0, 1]  # Full height of the plot

    line.set_data(x, y)
    line_V.set_data(x, y_V)
    vertical_line.set_data([x_vertical[0], x_vertical[0]], y_vertical)  # Plot as a vertical line
    return line, line_V, vertical_line

# Create the animation
anim = animation.FuncAnimation(
    fig, animate, init_func=init_lines, frames=selected_frames, interval=40, blit=True
)

# Save the animation as a GIF
writer = animation.PillowWriter(fps=1, metadata=dict(artist='Me'), bitrate=1800)
anim.save('/home/albertos/quantumInfo/ex5/real_space.gif', writer=writer)


plt.show()



#############################################################################################
#Repeat the same operations but this time for different values of T