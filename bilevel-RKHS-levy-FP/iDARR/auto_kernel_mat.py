import numpy as np

def auto_kernel_mat(regressionData, rkhs_type, varargin = 0):
    """
    Construct from data the evaluation of kernel, basis functions and kernel matrix.
    Inputs:
       regressionData: discrtized data from the inverse problem model
       rhks_type: type of rkhs kernel
           'auto': data-adaptive RKHS kernel for auto regularization
           'gauss': Gaussian kernel
       varargin:
           rkhs_type='gauss', p = varargin(1)---the bandwidth for the Gaussian kernel)
    Outputs:
          Gbar_D: matrix of evaluations of the kernel on {s_l}---nsxns
          basis_D: matrix of values of n0xJ basis functions on {s_l} points--- n0Jxns
          Sigma_D: kernel matrix via n0J linear functionals---n0Jxn0J
    
    """
    if rkhs_type == 'gauss':
        l = varargin
    elif rkhs_type == 'auto':
        pass  # Do nothing, just a placeholder
    else:
        raise ValueError('Wrong RKHS kernel type')

    r_seq = regressionData['r_seq']  # when r_seq is non-uniform, use dr = r_seq(2:end) - r_seq(1:end-1).       
    dx = r_seq[1] - r_seq[0]
    ds = dx
    rho_val = regressionData['rho_val']

    g = regressionData['g_ukxj']
    ns, J, n0 = g.shape
    k = n0 * J
    g1 = np.zeros((ns, k))

    for i in range(n0):
        g1[:, i * J:(i + 1) * J] = g[:, :, i]

    g1 = g1.T  # Transpose to shape (n0J, ns)

    if rkhs_type == 'auto':
        Gbar = regressionData['Gbar']
        Gbar_D = Gbar / (rho_val[:, None] * rho_val[None, :])
    elif rkhs_type == 'gauss':
        # l = 0.01
        G_fun = lambda s1, s2: Gauss(s1, s2, l)
        rr1, rr2 = np.meshgrid(r_seq, r_seq)
        G_mat = np.vectorize(G_fun)(rr1, rr2)
        Gbar_D = G_mat

    basis_D = g1 @ Gbar_D * ds  # (k,ns)x(ns,ns)-->(k,ns)
    Sigma1 = g1 @ Gbar_D @ g1.T  # kxk
    Sigma_D = Sigma1 * ds**2

    return Gbar_D, basis_D, Sigma_D

def Gauss(s1, s2, l):
    d = (s1 - s2) ** 2
    val = np.exp(-d / (2 * l))
    return val

