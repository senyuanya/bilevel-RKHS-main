import numpy as np
import random
import matplotlib.pyplot as plt
from scipy.stats import levy_stable
from generateData.generateSDEDataPureJump import simulate_SDEPureJump

# Set random number seed
seed = 0
np.random.seed(seed)

noise_type = "laplacejump"#"compoundpoisson"
initial_type = "zero" #"zero"; "piecewise" 
#Set parameters
T = 5  # total time
#X0 = 0 # initial value
Nx = 1000000 # # number of trajectories to generate
# # Initial Gaussian distribution
# mu0, sigma0 = 0.0, 1/np.sqrt(80)
# X0 = np.random.normal(mu0, sigma0, size=Nx)

dt =  0.05
Nt = int(T / dt) # number of time point to generate


#Obtain the SDE trajectory data
#Simulate compound poisson
def simulate_compound_poisson(lambda_param, dt, Nx, Nt):
    jumps = np.zeros((Nx, Nt))  # Initialize an array for jumps
    
    for i in range(Nx):
        # Simulate the number of jumps at each time step
        N_t = np.random.poisson(lambda_param * dt, size=Nt)
        
        # Iterate through each time step to sum jump sizes
        for t in range(Nt):
            if N_t[t] > 0:  # Only generate jump sizes if there are jumps
                jump_sizes = np.random.normal(0, np.sqrt(1/2), size=N_t[t])
                jumps[i, t] = np.sum(jump_sizes)  # Sum jumps for this time step

    return jumps

def simulate_laplace(lambda_param, dt, Nx, Nt):
    """
    Simulate increments ΔZ_{t_i} of a compound Poisson Lévy process with Laplace jumps.
      - Jump arrival:   N_{Δt} ~ Poisson(rate * dt)
      - Jump sizes:     Y ~ Laplace(loc=0, scale=1/laplace_lambda), symmetric
    """

    jumps = np.zeros((Nx, Nt))
    
    laplace_lambda=2.0
    scale = 1.0 / laplace_lambda  # Laplace scale b

    for i in range(Nx):
        # Poisson counts for each (path, time step)
        N_t = np.random.poisson(lambda_param * dt, size=Nt)
        for t in range(Nt):
            if N_t[t] > 0:
                # Laplace(0, b) with b=1/lambda; sum of n jumps in this interval
                jump_sizes = np.random.laplace(loc=0.0, scale=scale, size=N_t[t])
                jumps[i, t] = np.sum(jump_sizes)
    return jumps

if noise_type == "laplacejump":
    lambda_param = 1.0  
    # Simulate Lévy motion increments
    Lt = simulate_laplace(lambda_param, dt, Nx, Nt)
    
elif noise_type == "compoundpoisson":
    alpha = 0
    lambda_param = np.sqrt(np.pi) 
    Lt = simulate_compound_poisson(lambda_param, dt, Nx, Nt)

else:
    print("Other noise type")
    
from generateInitialData import generate_initialsamples, generate_piecewise_constant

if initial_type == "random":
    # Generate path
    X0 = generate_initialsamples(Nx, 0.01, jump_prob=0.01, big_scale=6.0)
    
    # Plot
    plt.figure(figsize=(12,6))
    plt.plot(np.linspace(0,1,Nx), X0, alpha=0.8)
    plt.title(r"Sample $X_0$ with intermittent big jumps", fontsize=14)
    plt.xlabel("Index", fontsize=12)
    plt.ylabel(r"$X_0$", fontsize=12)
    plt.grid(True)
    plt.show()
elif initial_type == "piecewise":
    segments = [250_000, 500_000, 750_000]
    values = [3, -1, 0, 2]
    X0 = generate_piecewise_constant(Nx, segments, values)
    # Plot
    plt.figure(figsize=(12,6))
    plt.plot(np.linspace(0,1,Nx), X0, alpha=0.8)
    plt.title(r"Piecewise constant step function $X_0$", fontsize=14)
    plt.xlabel("Index", fontsize=12)
    plt.ylabel(r"$X_0$", fontsize=12)
    plt.grid(True)
    plt.show()
else:
    X0 = 0

#================== Explicit Euler method =======================
Dataset_Explicit, trajectories_Explicit, timeset_Explicit = simulate_SDEPureJump(dt, T, Nx, X0, Lt, noise_type, initial_type, method='Explicit')

from generateData.generatePDFdataKDE import genereatePDFdataKDE, save_densitydata

#================== KDE method (Save PDF Data) =======================
#Obtain PDF data
bandwidth = 0.021 #0.105#0.21#0.021
x_grid = np.linspace(-5, 5, 1000)
pdf_original, pdf_data = genereatePDFdataKDE(trajectories_Explicit, x_grid, bandwidth)

time_grid = np.linspace(0, T, Nt + 1)
save_densitydata(dt, T, x_grid, time_grid, pdf_original, pdf_data, bandwidth, noise_type, initial_type, method='Explicit')

#Compare the nonlocal term
from generateData.generateDataKDE import compute_nonlocal_term_symmetric_convolution, compute_FDM_purejump
supp_H = [0, 2]
delta = round(supp_H[-1], 3)
dx = x_grid[1]-x_grid[0]
bdry_width = int(np.floor((delta+ 1e-10) / dx ))  # boundary space for interaction range with kernel
r_seq = dx * np.arange(1, bdry_width + 1)
if noise_type == "laplacejump":
    K_true = lambda r: 1*np.exp(-2*r)
elif noise_type == "compoundpoisson":
    K_true = lambda r: np.exp(-r**2)
fx_vals, valid_indices = compute_nonlocal_term_symmetric_convolution(pdf_data, x_grid, K_true, bdry_width)
x_indices_inuse = np.arange(bdry_width, len(x_grid) - bdry_width)

p_val, dp_dt_FDM = compute_FDM_purejump(pdf_data, x_indices_inuse, dt)

# Plot comparison
t_idx = 50
plt.figure(figsize=(10, 6), dpi = 500)
plt.plot(x_grid[valid_indices], dp_dt_FDM[t_idx,valid_indices], color='#808000', linestyle='-.', label= r"$\partial_t p(x,t)$ FDM")
plt.plot(x_grid[valid_indices], fx_vals[t_idx,valid_indices], 'r--', label=r"Nonlocal RHS: $\int [p(x - y) - p(x)] \phi(y)\,dy$")
plt.title(f"Verification of the Fokker-Planck Equation (time = {time_grid[t_idx]})")
plt.xlabel("$x$")
plt.ylabel("Value")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()