import numpy as np
import random
import matplotlib.pyplot as plt
from scipy.special import gamma
import add_mypathlevy

from generateData.generatePDFdataKDE import load_densitydata

# Seperate the data
def select_interval_points(arr, num_points):
    # Generate evenly spaced values from 1 to len(arr), round them, and subtract 1 to convert to zero-based indices
    indices = np.round(np.linspace(1, len(arr), num_points)).astype(int) - 1
    return arr[indices]


# Compute the nonlocal term
def compute_nonlocal_term_symmetric_convolution(u_vals, x_mesh, K_true, bdry_width):
    """
    Compute the nonlocal operator:
        fx(x) = ∫ [u(x+y) + u(x−y) − 2u(x)] * K_true(y) dy
    using symmetric convolution around each interior grid point.
    """

    dx = x_mesh[1] - x_mesh[0]
    Nx = len(x_mesh)
    Nt = u_vals.shape[0]
    x_indices = np.arange(bdry_width, Nx - bdry_width)

    # Relative offsets for convolution
    r_seq = np.arange(1, bdry_width + 1) * dx
    g_nu_values = np.array([K_true(r) for r in r_seq])

    # Define symmetric finite-difference function
    fun_g_vec = lambda u, xInd, rInd: u[xInd + rInd] - u[xInd]

    # Preallocate result
    fx_vals = np.zeros_like(u_vals)

    for n in range(Nt):
        u_snapshot = u_vals[n]
        for idx in x_indices:
            ind_p = np.arange(1, bdry_width + 1)
            ind_m = -np.arange(1, bdry_width + 1)
            temp_p = fun_g_vec(u_snapshot, idx, ind_p)
            temp_m = fun_g_vec(u_snapshot, idx, ind_m)
            conv = temp_p + temp_m
            fx_vals[n, idx] = np.dot(g_nu_values, conv) * dx

    return fx_vals, x_indices


## Finite difference method
# Storage for the nonlocal term:
def compute_FDM_purejump(density_val, x_indices, dt):
    """
    Compute the forward time difference ∂p/∂t for given density data.
    """
    N_x = density_val.shape[1]
    N_t = density_val.shape[0]
    f_val = np.zeros((N_t, N_x))
    p_val = np.zeros((N_t, N_x))
    for n in range(N_t - 1):
        p_current = density_val[n, :]
        p_next = density_val[n+1, :]
        f_data = np.zeros_like(p_current)
        for idx in x_indices:
            dp_dt = (p_next[idx] - p_current[idx]) / dt
            f_data[idx] = dp_dt
        
        f_val[n+1,:] = f_data
        p_val[n+1,:] = p_next
    return p_val, f_val

# Obtain PDF data and compute ∂_t p(x,t) from data 
def generateDataKDE(t_range, dt, N, r0, bandwidth, kernel_type, initial_type, example_type):
    xDim = 1  # spatial dimension; bandwidth = 0.21
    
    # === 1. Set the nonlocal operator type ===
    temp = example_type.split('_')
    example_type_base = temp[0]

    if example_type_base == 'nonlocal':
        fun_g = lambda u, x, y: u(x + y) - u(x)
        fun_g_vec = lambda u, xInd, rInd: u[xInd + rInd] - u[xInd]
    elif example_type_base == 'LinearIntOpt':
        fun_g = lambda u, Du, x, y: u(x + y)
        fun_g_vec = lambda u, Du, xInd, rInd: u[xInd + rInd]
    elif example_type_base == 'nonlinearOpt':
        fun_g = lambda u, Du, x, y: Du(x + y) * u(x)
        fun_g_vec = lambda u, Du, xInd, rInd: Du[xInd + rInd] * u[xInd]
    else:
        raise ValueError("Unsupported example_type. Use 'nonlocal', 'LinearIntOpt', or 'nonlinearOpt'.")
    
    obsInfo = {
        'example_type': example_type,
        'fun_g': fun_g,
        'fun_g_vec': fun_g_vec
    }
    
    # === 2. Define the kernel K_true(r) ===
    
    kernelInfo = {'d': xDim}
    if kernel_type == 'Compoundlevy':
        # In our problem, we require g_nu(r)=exp(-r^2)
        K_true = lambda r: np.exp(-r**2)
        kernel_str = 'Compoundlevy'
        threshold = 1e-10
    elif kernel_type == 'Laplacejump':
        K_true = lambda r: 1*np.exp(-2*r)
        kernel_str = 'Laplacejump'
        threshold = 1e-10
    elif kernel_type == 'Gaussian':
        s = 0.75
        mu = 0
        K_true = lambda r: np.exp(-0.5*((r - mu)/s)**2) / (np.sqrt(2*np.pi)*s)
        kernel_str = f'{kernel_type}_mean_{mu}_std_{s}'
        threshold = 1e-10
    elif kernel_type == 'Cauchy':
        K_true = lambda r: (1/np.pi) * ((1/r**2) * (r > 0.05) + (1/0.05**2) * (r <= 0.05))
        kernel_str = 'Cauchy'
        threshold = 1e-4
    elif kernel_type == 'Stablelevy':
        alpha_val = 1.5
        term1 = alpha_val / (2**(1 - alpha_val) * np.sqrt(np.pi))
        term2 = gamma((1 + alpha_val) / 2) / gamma(1 - alpha_val / 2)
        c_alpha = term1 * term2
        K_true = lambda r: c_alpha * np.where(r > r0, 1 / r**(1 + alpha_val), 1 / r0**(1 + alpha_val))
        kernel_str = 'Stablelevy'
        threshold = 1e-4
    else:
        raise ValueError("Unsupported kernel_type. Choose from 'Compoundlevy', 'Laplacejump', 'Gaussian', 'Cauchy', or 'Stablelevy'.")
    
    kernelInfo.update({
        'K_true': K_true,
        'kernel_type': kernel_type,
        'kernel_str': kernel_str
    })
    obsInfo['threshold'] = threshold
    
    # === 3. Set up the spatial and temporal meshes ===
    #Load PDF data
    if kernel_type == 'Compoundlevy':
        noise_type = "compoundpoisson"
    elif kernel_type == 'Laplacejump':
        noise_type = "laplacejump"
    else:
        raise ValueError("Unsupported kernel_type. Choose from 'Compoundlevy'.")
    T = t_range[1]    
    p_set, param_set = load_densitydata(dt, T, bandwidth, noise_type, initial_type, method='Explicit')
    
    # Spatial grid:
    x_mesh = param_set["x_grid"]
    x_min, x_max = x_mesh[0], x_mesh[-1]
    dx = x_mesh[1] - x_mesh[0]
    x_mesh_str = f"{x_min}_{dx}_{x_max}".replace('.', '_')
    obsInfo['x_mesh_data'] = x_mesh
    obsInfo['x_mesh_dx'] = dx
    obsInfo['x_mesh_str'] = x_mesh_str
    
    # Temporal grid:
    t_mesh = param_set["time_grid"]
    t_min, t_max = t_range
    obsInfo['t_mesh_data'] = t_mesh
    obsInfo['t_mesh_dt'] = dt
    
    # == 4. Set the grid of r ===
    # Assume that dr can be different from dx, and the integration interval is [0,2]
    supp_H = [0, 2]
    obsInfo['supp_H'] = supp_H
    obsInfo['delta'] = round(supp_H[-1], 3)
    print(f'\nKernel support range using for generating p(x,t): [{supp_H[0]:.2f}, {supp_H[1]:.2f}]\n')
    bdry_width = int(np.floor((obsInfo['delta']  + 1e-10) / dx))
    Index_xi_inUse = np.arange(bdry_width, len(x_mesh) - bdry_width)
    
    # === 5. Compute ∂_t p(x,t) directly with Finite difference method ===
    ## Finite difference method
    p_FDM, dp_dt_FDM = compute_FDM_purejump(p_set, Index_xi_inUse, dt)
    
    # 5. Uniformly sample N time slices (from the second half of the simulation)
    # sample_indices = np.linspace(N_tc // 2, N_tc - 1, N, dtype=int)
    sample_indices_matlab = np.round(np.linspace(10, 99, N)).astype(int)
    sample_indices = sample_indices_matlab - 1
    train_ind = select_interval_points(sample_indices, int(N / 2))
    test_ind = np.setdiff1d(sample_indices, train_ind)
    obsInfo['train_ind'] = train_ind
    obsInfo['test_ind'] = test_ind
    
    # Create train and test datasets for the PDF and the recomputed nonlocal term:
    ux_valtr = p_FDM[train_ind, :]
    fx_valtr = dp_dt_FDM[train_ind, :]
    ux_valts = p_FDM[test_ind, :]
    fx_valts= dp_dt_FDM[test_ind, :]
    
    return obsInfo, kernelInfo, ux_valtr, fx_valtr, ux_valts, fx_valts