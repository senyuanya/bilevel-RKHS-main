import numpy as np
from generateData.normalize_byL2norm import normalize_byL2norm 

def getData4regression_auto(ux_val, fx_val, ux_val_ts, fx_val_ts, dx, obsInfo, bdry_width, Index_xi_inUse, r_seq, data_str, normalizeOn):
    """
    pre-processData for regression: 
                min |L_phi[u]- f|^2 = c'*Abar*c - 2c'*b + bnorm_sq  
     
    L_phi[u](x) = \int phi(|y|)*g[u](x,y) dy 
                = \sum_r phi(r)* [ g[u](x,x+r) + g[u](x,x-r) ] dr 
     1. g[u](x,y) =  u(x+y)          linearIntOpt: integral kernel: 1D, Laplace Transform 
     2. g[u](x,y) =  [u(x+y)-u(x)]   nonlocal kernel
     3. g[u](x,y) = D[u(x+y))u(x)]   mfOpt: particle interaction --- 1D, div(u(x+y)) = u'(x+y) 
    
    ====== In the inverse-integral-operator
      <L_phi[u], L_psi[u]> = \int_x L_phi[u](x) L_psi[u](x) dx
                           = \sum_{r,s} phi(r)psi(s) \int_x convl_gu(r,x)convl_gu(s,x)dx drds
             convl_gu(r,x) = [ g[u](x,x+r) + g[u](x,x-r) ]          ***** to be computed here
                    G(r,s) = \int_x convl_gu(r,x)convl_gu(s,x)dx    *****
     - for function learning: we need the
                    A(i,j) = <L_{phi_i}[u], L_{phi_j}[u]> = sum_{r,s} phi_i(r)phi_j(s) G(r,s) drds
                    b(i)   = <L_{phi_i}[u], f> = int_{r} phi_i(r) sum_x convl_gu(r,x)f(x)dx  dr
                           >>>>>                                  *** convl_gu_f(r)   *****
     
     - for vector learning: we only need [with phi_i being the indicator at r_i ]
                    A(i,j) = G(r_i,r_j) *dr*dr
                    b(i)   = G*phi  = convl_gu_f(r_i) *dr
    == Output: ===>>>  ***** in either case, we only need those *****
    - compute  convl_gu(r,x) = [ g[u](x,x+r) + g[u](x,x-r) ] -- used in G, rho.           ----It is denoted by g_ukxj(s):=g[u_k](x_j,s) in the paper. 
    - compute  G(r,s)    --- used in vector & function learning
    - compute  rho(r)    --- used in all:  exploration measure rho(r) = \int   |convl_gu(r,x)| dx

    """
    
    # 1. Load and normalize process data (u, f)
    ux_shape = ux_val.shape
    ux_shape_ts = ux_val_ts.shape
    
    if len(ux_shape) == 3:  # u(n,t,x)
        case_num, T, x_num = ux_shape
        N = case_num * T
        ux_val = ux_val.reshape(case_num * T, x_num)
        fx_val = fx_val.reshape(case_num * T, x_num)
    elif len(ux_shape) == 2:  # u(n,x)
        case_num, x_num = ux_shape
        N = case_num
        T = 1
        
    if len(ux_shape_ts) == 3:  # u(n,t,x)
        case_num_ts, T_ts, x_num_ts = ux_shape_ts
        N_ts = case_num_ts * T_ts
        ux_val_ts = ux_val.reshape(case_num_ts * T_ts, x_num_ts)
        fx_val_ts = fx_val.reshape(case_num_ts * T_ts, x_num_ts)
    elif len(ux_shape_ts) == 2:  # u(n,x)
        case_num_ts, x_num_ts = ux_shape_ts
        N_ts = case_num_ts
        T_ts = 1

    # Normalize by L2 norm if normalizeOn
    if normalizeOn:
        ux_val, fx_val = normalize_byL2norm(ux_val, fx_val, dx)
        ux_val_ts, fx_val_ts = normalize_byL2norm(ux_val_ts, fx_val_ts, dx)
        data_str += 'NormalizeL2'
    
    N_xi_inUse = x_num - 2 * bdry_width
    N_r_seq = len(r_seq)  #  r_seq       = dx*(1:bdry_width);
    
    if (len(Index_xi_inUse) != N_xi_inUse) or (N_r_seq != bdry_width and bdry_width > 0):
        raise ValueError('Index_xi_inUse does not match with data and bdry_width.')  # error and terminate
    
    # 2. Get data for regression
    # get data for regression: convl_gu(r,x) = g[u](x,x+r) + g[u](x,x-r);
    fun_g_vec = obsInfo['fun_g_vec']
    ind_p = np.arange(1, bdry_width + 1)
    ind_m = -np.arange(1, bdry_width + 1) # Index plus r  

    g_ukxj = np.zeros((N_r_seq, N_xi_inUse, N)) # the array: g_kjl = g[u_k](x_j,r_l)
    g_ukxj_ts = np.zeros((N_r_seq, N_xi_inUse, N)) # the array used for testing: g_ukxj_ts = g[u_k](x_j,r_l)
    rhoN = np.zeros((N_r_seq, N))  # N copies: exploration measure rho: rho(r) = \int   |convl_gu(r,x)| dx
    rhoN_ts = np.zeros((N_r_seq, N))  # Testing N copies: exploration measure rho: rho(r) = \int   |convl_gu(r,x)| dx
    GN = np.zeros((N_r_seq, N_r_seq, N))  # N copies: G(r,s) = \int_x convl_gu(r,x)convl_gu(s,x)dx 
    gu_fN = np.zeros((N_r_seq, N))  #  N copies: sum_x convl_gu(r,x)f(x)dx
    gu_fN_ts = np.zeros((N_r_seq, N))  #  N copies: sum_x convl_gu(r,x)f_ts(x)dx
    gu_fN2 = np.zeros_like(gu_fN)  # a downsampled estimator of gu_fN 
    gu_fN2_ts = np.zeros_like(gu_fN_ts)  # a downsampled estimator of gu_fN 
    bnorm_sq = 0
    bnorm_sq_ts = 0
    fx_vec = np.zeros((N, N_xi_inUse))  # right-hand side of f_i with values on Index_xi_inUse
    fx_vec_ts = np.zeros((N, N_xi_inUse))  # right-hand side of f_i with testing values on Index_xi_inUse
    
    for nn in range(N):  # compute these terms for each u(x) 
        u1 = ux_val[nn, :]
        f1 = fx_val[nn, Index_xi_inUse]
        fx_vec[nn, :] = f1
        ##validation or test data
        u1_ts = ux_val_ts[nn, :]
        f1_ts = fx_val_ts[nn, Index_xi_inUse]
        fx_vec_ts[nn, :] = f1_ts

        convl_gu = np.zeros((N_r_seq, N_xi_inUse))
        val_abs = np.zeros_like(convl_gu)
        ###validation
        convl_gu_ts = np.zeros((N_r_seq, N_xi_inUse))
        val_abs_ts = np.zeros_like(convl_gu_ts)
        
        for k in range(N_xi_inUse):
            temp_p = fun_g_vec(u1, k + bdry_width, ind_p)
            temp_m = fun_g_vec(u1, k + bdry_width, ind_m)
            convl_gu[:, k] = temp_p + temp_m
            val_abs[:, k] = np.abs(temp_p) + np.abs(temp_m)
            ##Validataion 
            temp_p_ts = fun_g_vec(u1_ts, k + bdry_width, ind_p)
            temp_m_ts = fun_g_vec(u1_ts, k + bdry_width, ind_m)
            convl_gu_ts[:, k] = temp_p_ts + temp_m_ts
            val_abs_ts[:, k] = np.abs(temp_p_ts) + np.abs(temp_m_ts)
            
        
        g_ukxj[:, :, nn] = convl_gu  # nsxJ
        rhoN[:, nn] = np.mean(val_abs, axis=1)
        GN[:, :, nn] = convl_gu @ convl_gu.T
        gu_fN[:, nn] = convl_gu @ f1
        bnorm_sq += np.sum(f1**2) * dx
        gu_fN2[:, nn] = convl_gu[:, ::2] @ f1[::2]
        ##Validation
        g_ukxj_ts[:, :, nn] = convl_gu_ts  # nsxJ
        rhoN_ts[:, nn] = np.mean(val_abs_ts, axis=1)
        gu_fN_ts[:, nn] = convl_gu_ts @ f1
        bnorm_sq_ts += np.sum(f1_ts**2) * dx
        gu_fN2_ts[:, nn] = convl_gu_ts[:, ::2] @ f1_ts[::2]

    rho_val = np.mean(rhoN, axis=1)
    indr = np.where(rho_val > 0)[0]
    rho_val = rho_val[indr] # remove those zero weight points 
    r_seq = r_seq[indr]

    regressionData = {}
    regressionData['rho_val'] = rho_val / np.sum(rho_val) / dx  # exploration measure  
    regressionData['Gbar'] = np.mean(GN, axis=2)[indr, :][:, indr] * dx  # integral kernel in the operator L_G     *dr*ds,  ---- FL0725: TBD Gbar./(rho_val*rho_val') 
    regressionData['gu_f'] = np.mean(gu_fN, axis=1)[indr] * dx  # b
    regressionData['gu_f2'] = np.mean(gu_fN2, axis=1)[indr] * dx  # b2
    # regressionData['g_ukxj'] = g_ukxj[indr, :, :]  #  remove those values of g with zero weight points ?
    regressionData['bnorm_sq'] = bnorm_sq / N
    regressionData['bdry_width'] = bdry_width
    regressionData['r_seq'] = r_seq
    regressionData['data_str'] = data_str
    # regressionData['fx_vec'] = fx_vec  # Noisy right-hand side, normalized. Is it right?
    ##Validation
    regressionData['gu_fN_ts'] = np.mean(gu_fN_ts, axis=1)[indr] * dx  # b
    regressionData['gu_fN2_ts'] = np.mean(gu_fN2_ts, axis=1)[indr] * dx  # b2
    # regressionData['g_ukxj_ts'] = g_ukxj_ts[indr, :, :]  #  remove those values of g with zero weight points ?
    regressionData['bnorm_sq_ts'] = bnorm_sq / N
    # regressionData['fx_vec_ts'] = fx_vec_ts
    #--------------------Levy---------------------------------------------------------
    regressionData['g_ukxj'] = (1 / np.sqrt(N)) * g_ukxj[indr, :, :]  * np.sqrt(dx)
    regressionData['fx_vec'] = (1 / np.sqrt(N)) * fx_vec * np.sqrt(dx)
    
    regressionData['g_ukxj_ts'] = (1 / np.sqrt(N)) * g_ukxj_ts[indr, :, :]  * np.sqrt(dx)
    regressionData['fx_vec_ts'] = (1 / np.sqrt(N)) * fx_vec_ts * np.sqrt(dx)

    return regressionData

