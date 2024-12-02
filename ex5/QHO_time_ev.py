import numpy as np
from math import factorial
import matplotlib.pyplot as plt
from matplotlib import animation
from numpy.linalg import norm
import math

import debugger_module as dbg
from matplotlib import rcParams

color_cycle = plt.cm.Set1.colors  # Setting a predefined colormap for the plots
plt.rcParams.update({'font.size': 24}) #setting font size for the plots

import numpy as np
from scipy.special import factorial

def hermite(x, n):
    """
    @brief Computes the Hermite polynomial of order n at given points x.
    
    The Hermite polynomials are defined recursively and are used in various quantum mechanical problems,
    including the quantum harmonic oscillator.
    
    @param x The points at which to evaluate the Hermite polynomial.
    @param n The order of the Hermite polynomial.
    
    @return The Hermite polynomial evaluated at x.
    """
    # Create a zero array of size n+1 to store the coefficients
    herm_coeffs = np.zeros(n+1)
    
    # Set the nth coefficient to 1 to define the polynomial
    herm_coeffs[n] = 1
    
    # Use numpy's polynomial function to evaluate the Hermite polynomial at x
    return np.polynomial.hermite.hermval(x, herm_coeffs)

def stationary_state(x, n, m):
    """
    @brief Returns the stationary state (wavefunction) of order n for the quantum harmonic oscillator.
    
    This function calculates the stationary state (wavefunction) for the quantum harmonic oscillator
    at a given position grid x. It uses the associated Hermite polynomial and a prefactor that depends
    on the mass of the particle and the order n.
    
    @param x The real-space grid at which to evaluate the wavefunction.
    @param n The order of the stationary state. n = 0 corresponds to the ground state.
    @param m The mass of the particle.
    
    @return The wavefunction for the stationary state of order n.
    """
    # Calculate the prefactor based on the mass m and the order n
    prefactor = 1./np.sqrt(2.**n * factorial(n)) * (m/(np.pi))**(0.25)
    
    # Calculate the wavefunction as the product of the prefactor, exponential, and Hermite polynomial
    psi = prefactor * np.exp(- m*x**2 / 2) * hermite(np.sqrt(m)*x, n)
    
    return psi

class Param:
    """
    @brief Container for holding all simulation parameters.
    
    This class stores all the parameters necessary for a quantum harmonic oscillator simulation,
    including spatial grid, time grid, and mass of the particle. It also stores whether to use
    imaginary time evolution or not.
    
    @param xmax The maximum value of the real-space grid, with the grid ranging from [-xmax, xmax].
    @param num_x The number of grid points in the real-space.
    @param T The total simulation time.
    @param m The mass of the particle.
    @param timesteps The number of time steps.
    @param im_time Whether to use imaginary time evolution (default is False).
    """
    def __init__(self, xmax: float, num_x: int, T: int, m: float, timesteps: int, im_time: bool = False) -> None:
        # Store parameters
        self.im_time = im_time   
        self.m = m

        # Space grid setup
        self.xmax = xmax
        self.num_x = num_x
        self.dx = 2 * xmax / num_x  # Space step size
        self.x = np.arange(-xmax + xmax / num_x, xmax, self.dx)  # Create position grid

        # Time grid setup
        self.T = T
        self.timesteps = timesteps
        self.dt = T / timesteps  # Time step size
        self.t = np.arange(0, T, self.dt)  # Create time grid

        # Momentum grid setup
        self.dk = np.pi / xmax  # Momentum step size
        # Create momentum grid by combining two ranges (positive and negative momenta)
        self.k = np.concatenate((np.arange(0, num_x / 2), np.arange(-num_x / 2, 0))) * self.dk


class Operators:
    """
    @brief Container for holding operators and wavefunction coefficients.
    
    This class holds the different operators and wavefunction coefficients used in the simulation,
    specifically for the time evolution operator, which is broken into the kinetic (K) and potential (R)
    parts as used in the Baker-Kennedy-Hausdorff (B-K-H) decomposition.
    
    @param res The resolution, i.e., the number of grid points in the discretization.
    """
    def __init__(self, res: int) -> None:
        # Initialize operators for potential, kinetic, and wavefunction components.
        self.R = np.empty(res, dtype=complex)  # Potential part of the time evolution operator (space domain)
        self.K = np.empty(res, dtype=complex)  # Kinetic part of the time evolution operator (momentum domain)
        self.wfc = np.empty(res, dtype=complex)  # Wavefunction coefficients storage


def init(par: Param, n: int, voffset: float = 0.0, wfcoffset: float = 0.0) -> Operators:
    """
    @brief Initializes the wavefunction coefficients and the potential for the simulation.

    This function sets up the initial state of the system by initializing the wavefunction and
    potential according to the quantum harmonic oscillator model. The wavefunction is initialized
    using the stationary state for a given eigenstate (determined by the order n of the Hermite polynomial).
    
    @param par The parameters of the simulation, including space, time, and mass.
    @param n The order of the Hermite polynomial (eigenstate level).
    @param voffset The offset of the potential in real space. Default is 0.
    @param wfcoffset The offset of the wavefunction in real space. Default is 0.
    
    @return An Operators object containing the initialized wavefunction and potential.
    """
    opr = Operators(len(par.x))

    # Set up the potential as a quadratic potential for the quantum harmonic oscillator
    opr.V = (0.5 / par.m) * (par.x - voffset) ** 2  # omega = 1 (standard for QHO)
    
    # Initialize the wavefunction using the stationary state of order n
    opr.wfc = stationary_state(par.x - wfcoffset, n, par.m).astype(complex)
    
    # Normalize the wavefunction
    opr.wfc /= np.sqrt(np.sum(np.abs(opr.wfc) ** 2) * par.dx)

    # Coefficient for the time evolution operator (imaginary time if requested)
    coeff = 1 if par.im_time else 1j

    # Momentum space time evolution operator (kinetic energy part)
    opr.K = np.exp(-(0.5 / par.m) * (par.k ** 2) * par.dt * coeff)
    
    # Real space time evolution operator (potential energy part)
    opr.R = np.exp(-0.5 * opr.V * par.dt * coeff)

    return opr

def average_position(par: Param, opr: Operators) -> float:
    """
    @brief Calculates the average position <Psi|x|Psi> of the particle.

    This function computes the expectation value of the position operator in the state
    given by the wavefunction |Psi>. It is calculated as the integral of the position weighted
    by the probability density.

    @param par The parameters of the simulation.
    @param opr The operators, including the wavefunction to calculate the expectation value.

    @return The average position of the particle.
    """
    # Wavefunction and its complex conjugate
    wfc_r = opr.wfc
    wfc_c = np.conj(wfc_r)

    # Position expectation value <x> = <Psi|x|Psi>
    x_avg = wfc_c * par.x * wfc_r

    # Integrate over the entire space to find the average position
    x_avg_final = sum(x_avg).real

    return x_avg_final * par.dx

def plot_frame(param, n_frame, res, avg_pos, n, m):
    """
    @brief Plots the wavefunction, potential, and average position at a given timestep.

    This function generates and saves a plot for the wavefunction and the potential at a specific
    timestep, as well as the average position of the particle.

    @param param The parameters of the simulation.
    @param n_frame The current frame (timestep).
    @param res The simulation results (wavefunction densities and potential).
    @param avg_pos The average position of the particle for each timestep.
    @param n The order of the Hermite polynomial for the stationary state.
    @param m The mass of the particle.
    """
    plt.figure(figsize=(8,8))
    
    # Plot the wavefunction and stationary state
    plt.plot(param.x, res[0, n_frame, :], lw=2, alpha=.7, linestyle='-', label='$\psi(x,t)$', color=color_cycle[0])
    plt.plot(param.x, np.abs(stationary_state(param.x, n=n, m=m))**2, lw=2, alpha=.7, linestyle='--', label='$\psi(x,0)$', color=color_cycle[1])
    
    # Plot the potential
    plt.plot(param.x, res[2, n_frame, :], label='V(x,t)', color=color_cycle[2], lw=2)
    
    # Plot the average position as a dotted line
    plt.axvline(avg_pos[n_frame], linestyle='dotted', color=color_cycle[4], lw=2, label=r"$\langle \psi | x | \psi \rangle$")
    
    plt.ylim([0, 1])
    plt.legend()
    plt.title(f'timestep = {n_frame}')
    plt.xticks(np.arange(-4, 5))
    plt.grid(True)
    plt.xlabel("Position (x)")
    plt.ylabel("Probability Density |ψ(x)|^2")
    
    # Save the plot
    if param.im_time:
        plt.savefig(f'/home/albertos/quantumInfo/ex5/plots/QHO_imtime_T{param.T}_frame{n_frame}.svg', format='svg', bbox_inches='tight')
    else:
        plt.savefig(f'/home/albertos/quantumInfo/ex5/plots/QHO_T{param.T}_frame{n_frame}.svg', format='svg', bbox_inches='tight')
    
    plt.show()

def split_op(par: Param, opr: Operators, q_0t_values: np.ndarray, plot_frames: bool, n: int, voffset: float = 0.0, debug: bool = False) -> None:
    """
    @brief Performs time evolution using the split-operator method.

    This method splits the Hamiltonian into kinetic and potential parts and evolves the wavefunction
    in both momentum and real space. The evolution is performed for each timestep, updating the wavefunction.

    @param par The parameters of the simulation.
    @param opr The operators of the simulation, including the wavefunction and potential.
    @param q_0t_values The trajectory of the particle's position for time-dependent potential.
    @param plot_frames Whether to plot the frames at each timestep.
    @param n The order of the Hermite polynomial for the stationary state.
    @param voffset The offset of the potential in real space.
    @param debug Whether to print debug information.
    
    @return None
        The function updates the `res` array in place, which stores the density and potential at each timestep.
    """
    # Initialize result arrays to store wavefunction densities and potential over time
    res, avg_position_values = np.zeros((3, par.timesteps + 1, par.num_x)), np.zeros((par.timesteps + 1))
    
    # Store initial wavefunction densities and potential
    res[0, 0, :] = np.abs(opr.wfc) ** 2
    res[1, 0, :] = np.abs(np.fft.fft(opr.wfc)) ** 2
    res[2, 0, :] = opr.V
    avg_position_values[0] = average_position(par, opr)

    # Check if q_0t_values has the correct shape
    if q_0t_values.shape[0] != par.timesteps:
        dbg.error(f"Invalid shape for q_0t_values: should have dimension {par.timesteps} while it has dimension {q_0t_values.shape[0]}")
    
    # Time evolution loop
    for i in range(1, par.timesteps):
        # Update the potential for the current timestep
        opr.V = (0.5 / par.m) * (par.x - voffset - q_0t_values[i]) ** 2
        
        # Update the real space time evolution operator
        coeff = 1 if par.im_time else 1j
        opr.R = np.exp(-0.5 * opr.V * par.dt * coeff)

        # Half-step in real space
        opr.wfc *= opr.R

        # FFT to momentum space
        opr.wfc = np.fft.fft(opr.wfc)

        # Full step in momentum space (kinetic evolution)
        opr.wfc *= opr.K

        # Inverse FFT back to real space
        opr.wfc = np.fft.ifft(opr.wfc)

        # Final half-step in real space
        opr.wfc *= opr.R

        # Calculate the density for plotting
        density = np.abs(opr.wfc) ** 2
        density_momentum = np.abs(np.fft.fft(opr.wfc)) ** 2

        # Normalize the wavefunction
        renorm_factor = np.sum(density * par.dx)
        opr.wfc /= np.sqrt(renorm_factor)

        # Store the wavefunction densities and potential for the current timestep
        res[0, i, :] = np.real(density)
        res[1, i, :] = np.real(density_momentum)
        res[2, i, :] = np.real(opr.V)
        avg_position_values[i] = average_position(par, opr)

    return res, avg_position_values

def calculate_energy(par: Param, opr: Operators) -> float:
    """
    @brief Calculate the total energy <Psi|H|Psi> using both real-space and momentum-space terms.

    This function calculates the total energy of the quantum system, which is the expectation value 
    of the Hamiltonian operator (H). The energy is computed by summing both the kinetic and potential 
    energy terms. The kinetic energy is evaluated in momentum space, while the potential energy is evaluated 
    in real space.

    @param[in] par The parameters for the simulation, including grid spacing, momentum values, and mass.
    @param[in] opr The operators and wavefunctions, including the wavefunction in real space and the potential.

    @return The total energy of the system, calculated as the expectation value <Psi|H|Psi>. The energy is 
            normalized by the grid spacing `par.dx`.
    
    @note The code assumes that the mass is normalized to 1. If you want to change the mass, modify 
          the appropriate terms in the kinetic energy calculation.
    """    # Creating real-space, momentum-space, and conjugate wavefunctions
    wfc_r = opr.wfc  # Wavefunction in real space
    wfc_k = np.fft.fft(wfc_r)  # Fourier transform of the wavefunction (momentum space)
    wfc_c = np.conj(wfc_r)  # Complex conjugate of the wavefunction in real space

    # Calculate the kinetic energy term (momentum space)
    # The energy in momentum space: T = p^2 / (2m), where p is the momentum
    # We assume m = 1 here. If not, divide by mass (par.m) appropriately.
    energy_k = 0.5 * wfc_c * np.fft.ifft((par.k ** 2) * wfc_k)  # Kinetic energy in real space

    # Calculate the potential energy term (real space)
    # V = 0.5 * m * x^2 for a harmonic oscillator potential, but we already have the potential V in `opr.V`
    energy_r = wfc_c * opr.V * wfc_r  # Potential energy in real space

    # Integrating over all space (in both real and momentum spaces)
    total_energy = sum(energy_k + energy_r).real  # Add the kinetic and potential energies

    # Return the energy normalized by the grid spacing
    return total_energy * par.dx



############################################################################################
##
## Given |Ψ0⟩ = |𝑛 = 0⟩ (ground state of the Harmonic oscillator), compute |Ψ(𝑡)⟩ for different values of 𝑇. 
## Plot the square norm of |Ψ(𝑡)⟩ as a function of 𝑞 at different times
## and the average position of the particle as a function of 𝑡.
##
def evolution_code(xmax, nx, tmax, nt, im_time, m, n, voffset, wfcoffset, debug, plot_frames, anim, frames=None):
    """
    @brief Simulates the time evolution of a quantum harmonic oscillator using the split-operator method.

    This function computes the time evolution of the wavefunction of a quantum harmonic oscillator
    using the split-operator method. It also includes options for visualizing the evolution, such as 
    plotting intermediate frames and generating animations of the wavefunction's evolution over time.

    @param[in] xmax The maximum value for the position space grid.
    @param[in] nx The number of grid points in position space.
    @param[in] tmax The total simulation time.
    @param[in] nt The number of time steps in the simulation.
    @param[in] im_time A boolean flag indicating whether to use imaginary time evolution.
    @param[in] m The mass of the particle (in normalized units).
    @param[in] n The quantum number of the initial state (n=0 for ground state).
    @param[in] voffset The offset applied to the potential.
    @param[in] wfcoffset The offset applied to the wavefunction.
    @param[in] debug A boolean flag to enable debug printing.
    @param[in] plot_frames A boolean flag to enable plotting of intermediate frames.
    @param[in] anim A boolean flag to enable the creation of an animation.
    @param[in] frames A list of frame indices to plot during the evolution (optional).

    @note The function uses the `split_op` method for time evolution and `plot_frame` to visualize 
          the wavefunction and average position at specific time steps.

    @return None The function performs the time evolution and generates visualizations but does not return a value.
    """
    # Initialize parameters for the simulation discretization
    params = Param(xmax=xmax, num_x=nx, timesteps=nt, T=tmax, im_time=im_time, m=m) 
    dbg.checkpoint(f"Time-step size: dt={round(params.dt,4)}", debug=debug)  # Debugging info for time step size
    q_0t_values = params.t / tmax  # Normalize the time values for plotting purposes

    # Initialize the wavefunction (at the ground state) and temporal evolution operators
    ops = init(params, n, voffset, wfcoffset)  # Initialize the operators and wavefunction for the QHO
    # Uncomment the following block to plot the initialized wavefunction and potential
    """
    plt.figure(figsize=(8,8))
    plt.plot(params.x, np.abs(ops.wfc)**2, lw=3, alpha=.5, linestyle='--', label='wfc')
    plt.plot(params.x, np.abs(stationary_state(params.x, n=n))**2, lw=3, alpha=.5, linestyle='-', label='stationary_state(n=0)')
    plt.plot(params.x,  ops.V, label='potential')
    plt.ylim([0, 1])
    plt.legend()
    plt.title('Initialized state')
    plt.show()
    """

    # Perform time evolution using the split-operator method
    res, avg_position_values = split_op(par=params, opr=ops, q_0t_values=q_0t_values, debug=debug, plot_frames=plot_frames, n=n)
    dbg.checkpoint("Evolution computed", debug=debug)  # Debugging info after evolution

    # Plot specific frames if requested
    if plot_frames and frames is not None:
        for fr in frames:
            plot_frame(params, n_frame=fr, res=res, avg_pos=avg_position_values, n=n, m=m)  # Plot intermediate time frames

    # If animation is enabled, create a time evolution animation
    if anim:
        # Select frames for animation (ensuring a reasonable number of frames)
        step_interval = max(params.timesteps // 200, 1)  # Ensure at least 200 frames
        selected_frames = range(0, params.timesteps + 1, step_interval)

        # Initialize the plot for the animation
        fig, ax = plt.subplots()
        ax.set_xlim(-params.xmax, params.xmax)
        ax.set_ylim(0, 1)
        ax.set_xlabel("Position (x)")
        ax.set_ylabel("Probability Density |ψ(x)|^2")

        # Create plot elements for potential, wavefunction, and average position
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

        # Animation function to update the plot at each frame
        def animate(i):
            x = params.x  # Position grid
            y = res[0, i, :]  # Wavefunction at the current time step
            y_V = res[2, i, :]  # Potential at the current time step

            # Vertical line representing the average position at the current time step
            x_vertical = [avg_position_values[i]]
            y_vertical = [0, 1]

            # Update the plot elements
            line.set_data(x, y)
            line_V.set_data(x, y_V)
            vertical_line.set_data([x_vertical[0], x_vertical[0]], y_vertical)  # Plot vertical line for average position
            return line, line_V, vertical_line

        # Create and save the animation as a GIF
        anim = animation.FuncAnimation(
            fig, animate, init_func=init_lines, frames=selected_frames, interval=40, blit=False
        )

        # Save the animation as a GIF file
        writer = animation.PillowWriter(fps=1, metadata=dict(artist='Me'), bitrate=1800)
        anim.save('/home/albertos/quantumInfo/ex5/real_space.gif', writer=writer)

        plt.show()


# Initializing variables
nx = 1000  # Number of grid points in position space (discretization of space)
xmax = 4  # Right boundary of the position space interval
nt = 30000  # Number of time steps (timesteps = T / dt)
m = 1  # Mass of the particle. Ensure it's correctly applied throughout the code if changing its value from 1

# Flags and options for simulation
im_time = True  # Boolean flag for imaginary-time evolution (if True, the evolution is done in imaginary time)
debug = True  # Enable debugging output (prints info at key points)
plot_frames = True  # Boolean flag to enable plotting intermediate frames of the wavefunction during evolution
frames = [1, 145, 250]  # Indices for specific frames to plot during evolution. For example, frames at indices 1, 145, and 250
# NOTE: At the index nt, the potential is zero; this could be a result of the potential settings or boundary conditions.

anim = False  # Boolean flag to enable animation of the wavefunction evolution

# Wavefunction setup
n = 4  # Quantum number for the energy level of the quantum harmonic oscillator (0 corresponds to the ground state)
wfcoffset = 0  # Offset for the wavefunction (shift of the initial wavefunction in position space)
voffset = 0  # Offset for the potential (shift of the potential energy function)

# Debugging message for variable initialization
dbg.checkpoint("Initialized variables", debug=debug)

# List of different values for the total simulation time (T)
T_values = [500, 2000, 5000, 10000, 20000, 30000, 40000]

# Loop over different simulation times (T values) and execute the time evolution for each one
for TT in T_values:
    dbg.checkpoint("--------------------", debug=debug)  # Separator for clarity in debugging output
    dbg.checkpoint(f"Execution T={TT}", debug=debug)  # Debugging message showing the current total time for evolution
    
    # Call the evolution_code function for each simulation time (T)
    # This will compute the time evolution of the wavefunction and plot the results for each case
    evolution_code(
        xmax=xmax,  # Right boundary of position space
        nx=nx,  # Number of grid points in position space
        nt=nt,  # Number of time steps
        tmax=TT,  # Total simulation time for this run
        im_time=im_time,  # Whether to use imaginary-time evolution
        m=m,  # Mass of the particle
        n=n,  # Energy level (quantum number) of the oscillator
        voffset=voffset,  # Offset for the potential
        wfcoffset=wfcoffset,  # Offset for the wavefunction
        debug=debug,  # Enable debug information
        plot_frames=plot_frames,  # Whether to plot intermediate frames
        anim=anim,  # Whether to generate an animation of the evolution
        frames=frames  # Specific frames to plot
    )