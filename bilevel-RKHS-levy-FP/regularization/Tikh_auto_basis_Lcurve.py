import numpy as np
from scipy.linalg import svd
from iDARR.auto_kernel_mat import auto_kernel_mat 
from regu_tools_Hansen.l_curvemy import l_curve,csvd
from regu_tools_Hansen.tikhonov import tikhonov
from scipy.io import savemat

def Tikh_auto_basis_Lcurve(regressionData, rkhs_type, *args):
    """
     First compute the SVD: Sigma_D = V*S*V^T, then determine the numerical
     rank of Sigma_D as r = length(ind) with  ind = find(s>tol).
     let x = Vr*y, then min{||Sigma_D*x-f||^2+lambda x^T*Sigma_D*x} becomes
     min{||Vr*Sr*y-f||^2+lambda ||Lr*y||^2}, whiich is equivalent to
     min{||Vr*Lr*z-f||^2+lambda ||z||^2}, where z=Lr*y
    """
    # Check for acceptable number of input arguments
    if len(args) < 1 and rkhs_type == 'Gaussian-RKHS':
        raise ValueError('Not Enough Inputs')

    if rkhs_type == 'auto-RKHS':
        _, basis_D, Sigma_D = auto_kernel_mat(regressionData, 'auto')
    elif rkhs_type == 'Gaussian-RKHS':
        l = args[0]
        _, basis_D, Sigma_D = auto_kernel_mat(regressionData, 'gauss', l)
    else:
        raise ValueError('Wrong RKHS kernel type')

    fx_vec = regressionData['fx_vec'].T    # Jxn0
    f = regressionData['fx_vec'].flatten()                   # n0J, vectorized from (1,{x_j}) to (n_0,{x_j})

    #-----------------------------------------------
    tol = 1e-3
    
    U, s, Vt = csvd(Sigma_D)
    V = Vt.T

    ind = np.where(s > tol)[0]   # numerical rank of Sigma_D
    r = len(ind)
    Vr = V[:, ind]      # compute an x in R(Vr)
    sr = s[ind]
    lr = np.sqrt(sr)
    
    # Find regularization parameter using L-curve
    # data_to_save = {
    # 'Vr': Vr,
    # 'lr': lr,
    # 'f': f
    #  }
    # savemat('reg.mat', data_to_save)
    lr = lr[:, np.newaxis]
    # print("Vr, lr, f:",Vr.shape, lr.shape, f.shape)
    reg_corner, _, _, _ = l_curve(Vr, lr, f, 'Tikh')
    z_reg, res, eta = tikhonov(Vr, lr, np.eye(r), f, reg_corner, np.zeros(r))
    # print("z_reg / lr:",z_reg / lr)
    # print("r:",r)
    
    c_reg = Vr @ (z_reg / lr)
    x_reg = basis_D.T @ c_reg

    return x_reg, res, eta, reg_corner

