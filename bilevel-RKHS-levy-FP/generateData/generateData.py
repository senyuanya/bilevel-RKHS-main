import numpy as np


# Seperate the data
def select_interval_points(arr, num_points):
    # Generate evenly spaced values from 1 to len(arr), round them, and subtract 1 to convert to zero-based indices
    indices = np.round(np.linspace(1, len(arr), num_points)).astype(int) - 1
    return arr[indices]


def generateObservations(obsInfo, u_val, N, factor_x, factor_t, scale_val, drift_func):
    """
    Subsample the dataset based on an integer multiple of the fine mesh size.
    This version vectorizes the spatial finite-difference computations.
    """

    # Retrieve fine-grid info
    t_mesh = obsInfo['t_mesh_data']
    dt = obsInfo['t_mesh_dt']
    x_mesh = obsInfo['x_mesh_data']
    dx = obsInfo['x_mesh_dx']
    
    # -------------------------------------------
    # 1. Extract the coarse meshes and corresponding data:
    p_val = u_val.copy()
    p_val_coarse = p_val[::factor_t, ::factor_x].copy()
    t_mesh_coarse = t_mesh[::factor_t]
    x_mesh_coarse = x_mesh[::factor_x]
    scale_coarse = scale_val[::factor_t]
    # New time and space steps on coarse grid:
    dt_new = factor_t * dt  
    dx_new = factor_x * dx  
    
    obsInfo['t_mesh_coarse'] = t_mesh_coarse
    obsInfo['x_mesh_coarse'] = x_mesh_coarse
    obsInfo['t_mesh_coarse_dt'] = dt_new
    obsInfo['x_mesh_coarse_dx'] = dx_new
    N_tc = len(t_mesh_coarse)
    N_xc = len(x_mesh_coarse)
    
    x_mesh_coarse_str = f"{x_mesh_coarse[0]}_{dx_new}_{x_mesh_coarse[-1]}".replace('.', '_')
    obsInfo['x_mesh_coarse_str'] = x_mesh_coarse_str
    
    # ----- Boundary point of generating PDF data -----------
    left_point = obsInfo['x_mesh_data'][obsInfo['Data_Index_xi_inUse'][0]]
    right_point = obsInfo['x_mesh_data'][obsInfo['Data_Index_xi_inUse'][-1]]
    
    
    # indice_left = np.where((x_mesh_coarse>= left_point-dx_new) & (x_mesh_coarse <= left_point+dx_new))[0]
    # indice_right = np.where((x_mesh_coarse>= right_point-dx_new) & (x_mesh_coarse <= right_point+dx_new))[0]
    
    indice_left = np.where( ((x_mesh_coarse >= left_point) & (x_mesh_coarse <= left_point + dx_new)) |
                              ((x_mesh_coarse > right_point) & (x_mesh_coarse <=  right_point + dx_new)) )[0]

    indice_right = np.where( ((x_mesh_coarse >= right_point - dx_new) & (x_mesh_coarse <= right_point)) |
                                ((x_mesh_coarse >= left_point - dx_new) & (x_mesh_coarse < left_point)) )[0]
    # print("Left :", indice_left)
    # print("Right :", indice_right)
    
    # -------------------------------------------
    # 2. Precompute the drift on the coarse grid
    # compute the drift function in advance
    f_x_coarse = np.array([drift_func(xi) for xi in x_mesh_coarse])
    
    # -------------------------------------------
    # 3. Recompute the nonlocal term on the coarse grid using vectorized spatial differences.
    N_t_coarse, _ = p_val_coarse.shape
    f_val_coarse_new = np.zeros_like(p_val_coarse)
    
    # Loop over coarse time steps (except the last, since we use forward difference in time)
    for n in range(N_t_coarse - 1):
        # p_current and p_next are 1D arrays at time n and n+1.
        p_current = p_val_coarse[n, :]
        p_next    = p_val_coarse[n+1, :]
        # For interior indices: 1 to N_xc-2
        # Compute advection using a central difference. Note: f(x) is replaced by precomputed f_x_coarse.
        adv = -( f_x_coarse[2:] * p_current[2:] - f_x_coarse[:-2] * p_current[:-2] ) / (2 * dx_new)
        # Compute diffusion using second-order central differences:
        diff = 0.5 * ( p_current[2:] - 2 * p_current[1:-1] + p_current[:-2] ) / (dx_new**2)
        # Time derivative (forward difference):
        dp_dt = ( p_next[1:-1] - p_current[1:-1] ) / dt_new
        # Update nonlocal term:
        f_val_coarse_new[n+1, 1:-1] = dp_dt - (adv + diff)
        
        # --- Boundary point processing ---
        # Left boundary: use forward differences
        for i_left in indice_left:
            # Advection term (forward difference):
            adv_left = -( -3 * f_x_coarse[i_left] * p_current[i_left] +
                          4 * f_x_coarse[i_left+1] * p_current[i_left+1] -
                          1 * f_x_coarse[i_left+2] * p_current[i_left+2] ) / (2 * dx_new)
            
            # Diffusion term (forward one-sided difference for second derivative):
            diff_left = 0.5 * ( 2 * p_current[i_left] -
                                5 * p_current[i_left+1] +
                                4 * p_current[i_left+2] -
                                p_current[i_left+3] ) / (dx_new**2)
            
            # Update the nonlocal term for the left boundary point:
            f_val_coarse_new[n+1, i_left] = ( (p_next[i_left] - p_current[i_left])/dt_new 
                                                  - (adv_left + diff_left) )
        
        # Right boundary: use backward differences
        for i_right in indice_right:
            # Advection term (backward difference):
            adv_right = -( 3 * f_x_coarse[i_right] * p_current[i_right] -
                          4 * f_x_coarse[i_right-1] * p_current[i_right-1] +
                          1 * f_x_coarse[i_right-2] * p_current[i_right-2] ) / (2 * dx_new)
            
            # Diffusion term (backward one-sided difference for second derivative):
            diff_right = 0.5 * ( 2 * p_current[i_right] -
                                  5 * p_current[i_right-1] +
                                  4 * p_current[i_right-2] -
                                  p_current[i_right-3] ) / (dx_new**2)
            
            # Update the nonlocal term for the right boundary point:
            f_val_coarse_new[n+1, i_right] = ( (p_next[i_right] - p_current[i_right])/dt_new 
                                                  - (adv_right + diff_right) )
    
    # -------------------------------------------
    # 4. Renormalize each time-slice so that the total probability is 1.
    # We assume scale_coarse is nonzero.
    # (This loop over time steps is inexpensive if the number of coarse time steps is small.)
    for n in range(N_tc):
        if scale_coarse[n] != 0:
            p_val_coarse[n, :] /= scale_coarse[n]
            f_val_coarse_new[n, :] /= scale_coarse[n]
    

    # -------------------------------------------
    # 5. Uniformly sample N time slices (from the second half of the simulation)
    # sample_indices = np.linspace(N_tc // 2, N_tc - 1, N, dtype=int)
    sample_indices_matlab = np.round(np.linspace(np.floor(N_tc/2) + 1, N_tc, N)).astype(int)
    sample_indices = sample_indices_matlab - 1
    train_ind = select_interval_points(sample_indices, int(N / 2))
    test_ind = np.setdiff1d(sample_indices, train_ind)
    obsInfo['train_ind'] = train_ind
    obsInfo['test_ind'] = test_ind
    
    # Create train and test datasets for the PDF and the recomputed nonlocal term:
    ux_valtr_coarse = p_val_coarse[train_ind, :]
    fx_valtr_coarse = f_val_coarse_new[train_ind, :]
    ux_valts_coarse = p_val_coarse[test_ind, :]
    fx_valts_coarse = f_val_coarse_new[test_ind, :]
    
    return obsInfo, ux_valtr_coarse, fx_valtr_coarse, ux_valts_coarse, fx_valts_coarse, p_val_coarse, f_val_coarse_new