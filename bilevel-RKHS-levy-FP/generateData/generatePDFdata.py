import numpy as np
from scipy.special import gamma


def generatePDFdata(x_range, t_range, dx, dt, r0, kernel_type, example_type, drift_func):
    """
    Generate the evolution of the PDF under the Fokker–Planck equation with Lévy noise.
    
    The discretization is
    \[
    p_i^{n+1} = p_i^n + \Delta t\Biggl[-\frac{f(x_{i+1})p_{i+1}^n - f(x_{i-1})p_{i-1}^n}{2\Delta x}
    + \frac{1}{2}\frac{p_{i+1}^n-2p_i^n+p_{i-1}^n}{\Delta x^2}
    + \sum_{k=1}^{M} e^{-r_k^2}\Bigl(p_{i+k}^n+p_{i-k}^n-2p_i^n\Bigr)\Delta r \Biggr],
    \]
    with \(g_\nu(r)=e^{-r^2}\). The nonlocal term is implemented via a convolution-like sum.
    
    The PDF is renormalized at every time step to maintain total probability equal to 1.
    """
    xDim = 1  # spatial dimension
    
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
    
    # Spatial grid:
    x_min, x_max = x_range
    x_points = int((x_max - x_min) / dx) + 1
    x_mesh = np.linspace(x_min, x_max, x_points)
    x_mesh_str = f"{x_min}_{dx}_{x_max}".replace('.', '_')
    obsInfo['x_mesh_data'] = x_mesh
    obsInfo['x_mesh_dx'] = dx
    obsInfo['x_mesh_str'] = x_mesh_str
    
    # Temporal grid:
    t_min, t_max = t_range
    t_points = int((t_max - t_min) / dt) + 1
    t_mesh = np.linspace(t_min, t_max, t_points)
    obsInfo['t_mesh_data'] = t_mesh
    obsInfo['t_mesh_dt'] = dt
    
    # === 4. Set the grid of r (non-local integration step) ===
    # Assume that dr can be different from dx, and the integration interval is [0,2]
    supp_H = [0, 2]
    obsInfo['supp_H'] = supp_H
    obsInfo['delta'] = round(supp_H[-1], 3)
    print(f'\nKernel support range using for generating p(x,t): [{supp_H[0]:.2f}, {supp_H[1]:.2f}]\n')
    bdry_width = int(np.floor((obsInfo['delta']  + 1e-10) / dx))
    Index_xi_inUse = np.arange(bdry_width, len(x_mesh) - bdry_width)
    r_seq = dx * np.arange(1, bdry_width + 1)
    # Define the offset index for the positive and negative directions
    ind_p = np.arange(1, bdry_width + 1)
    ind_m = -np.arange(1, bdry_width + 1)
        
    N_t = len(t_mesh)
    N_x = len(x_mesh)
    
    # === 5. Set the initial PDF ===
    # Use a normalized Gaussian PDF centered at 0 with standard deviation sigma0.
    mu0 = 0      
    # The inital PDF settting in accordance with paper: " Fokker–Planck equations for stochastic dynamical systems with symmetric Lévy motions"
    pdf_init = np.sqrt(40 / np.pi) * np.exp(-40 * x_mesh**2)

    # # Normalize just in case (should be close to 1)
    pdf_init /= np.trapz(pdf_init, x_mesh)
    
    # Set the PDF tensor
    p_val = np.zeros((N_t, N_x))
    p_val[0, :] = pdf_init
    # Number of interior indices 
    obsInfo['Data_Index_xi_inUse'] = Index_xi_inUse
    # Precompute the kernel for the jump measure:
    g_nu_values = np.array([K_true(r) for r in r_seq])
    dr = r_seq[1]-r_seq[0]
    
    # Storage for the nonlocal term:
    f_val = np.zeros((N_t, N_x))
    # Storage for the scale value:
    scale_val = np.zeros(N_t)
    scale_val[0] = np.trapz(pdf_init, x_mesh)
    
    # === 6. Solving the Fokker-Planck equation with Finite Difference Method ===

    for n in range(N_t - 1):
        p_prev = p_val[n, :]
        p_new = np.copy(p_prev)
        f_data = np.zeros_like(p_prev)

        for idx in Index_xi_inUse:
            # Compute the nonlocal term:
            temp_p = fun_g_vec(p_prev, idx, ind_p)
            temp_m = fun_g_vec(p_prev, idx, ind_m)
            convl_gu = temp_p + temp_m
            nonlocal_term = g_nu_values @ convl_gu * dr
            
            # Compute the drift (advection) term via central difference:
            advection = -(drift_func(x_mesh[idx + 1]) * p_prev[idx + 1] - drift_func(x_mesh[idx - 1]) * p_prev[idx - 1]) / (2 * dx)
    
            # Compute the diffusion term via a second-order central difference:
            diffusion = 0.5 * (p_prev[idx + 1] - 2 * p_prev[idx] + p_prev[idx - 1]) / (dx**2)
    
            # Update the PDF at the current spatial index:
            p_new[idx] = p_prev[idx] + dt * (advection + diffusion + nonlocal_term)
            
            # Store the nonlocal term (optional)
            f_data[idx] = nonlocal_term
            
        # Renormalize the PDF so that the total probability is 1.
        norm_val = np.trapz(p_new, x_mesh)
        scale_val[n+1] = norm_val
        p_val[n + 1, :] = p_new
        f_val[n + 1, :] = f_data

    return obsInfo, kernelInfo, p_val, f_val, scale_val

