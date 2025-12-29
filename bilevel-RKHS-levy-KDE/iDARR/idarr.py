import numpy as np
from iDARR.auto_kernel_mat import auto_kernel_mat
from iDARR.cg_rkhs import cg_rkhs
def idarr(regressionData, rkhs_type, K, stop_rule, nsr, varargin = 0):
    """
    Iterative data-adaptive RKHS regularization by CG.
 
      Inputs:
        regressionData: discrtized data from the inverse problem model
        rhks_type: type of rkhs kernel
            'auto': data-adaptive RKHS kernel for auto regularization
            'gauss': Gaussian kernel
        K: maximum iteration
        stop_rule:
            'DP': discrepancy principle
            'LC': L-curve
        nsr: noise norm. Set to 0 if unkown
        varargin:
            rkhs_type='gauss', p = varargin(1)---the bandwidth for the Gaussian kernel) 
    """

    if rkhs_type == 'auto':
        _, basis_D, Sigma_D = auto_kernel_mat(regressionData, 'auto')
    elif rkhs_type == 'gauss':
        if varargin == 0:
            raise ValueError('Missing argument for Gaussian kernel')
        else:
            l = varargin
            _, basis_D, Sigma_D = auto_kernel_mat(regressionData, 'gauss', l)
    else:
        raise ValueError('Wrong RKHS kernel type')

    fx_vec = regressionData['fx_vec'] #.T    # Jxn0
    f = fx_vec.ravel()                     # n0J, vectorized from (1,{x_j}) to (n_0,{x_j})

    if stop_rule == 'LC':
        nsr = 0  
        NoStop = 'on'  # no noise level provided--the iteration should run to complete and then use L-curve 
    elif stop_rule == 'DP':
        NoStop = 'off'

    C, res, eta, iter_stop = cg_rkhs(Sigma_D, f, K, stop_rule, nsr, NoStop)
    X = basis_D.T @ C  # (ns,n0J)x(n0J,k)-->(ns,k)

    return X, res, eta, iter_stop

