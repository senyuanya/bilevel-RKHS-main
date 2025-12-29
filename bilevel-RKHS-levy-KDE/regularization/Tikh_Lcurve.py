import numpy as np
from scipy.linalg import sqrtm
# from regu_tools_Hansen.lcurve_functions import l_cuve
from regu_tools_Hansen.l_curvemy import l_curve
from regu_tools_Hansen.cgsvd import cgsvd
from regu_tools_Hansen.tikhonov import tikhonov
from scipy.io import savemat

def Tikh_Lcurve(A, b, B, method):
    # Compute the square root of matrix B
    L = np.real(sqrtm(B))
    L = np.copy(L)
    # print("L.shape:",L.shape)
    # Perform the CG-SVD decomposition
    # data = {
    # 'A': A,
    # 'L':L
    # }
    # savemat('AL.mat', data)
    # np.savez('AL.npz', A=A, L=L)
    U, sm, X, _, _= cgsvd(A, L)
    # print("U, sm, X:",U, sm, X)
    # Initialize x0
    n = A.shape[1]
    x0 = np.zeros(n)

    # Compute the regularization parameter using the L-curve method

    reg_corner, res, eta, reg_param = l_curve(U, sm, b, method)  # method = 'tsvd', 'Tikh', 'dsvd'
    # print("reg_corner:",reg_corner)
    # Solve the Tikhonov regularization problem
    x_reg, res, eta = tikhonov(U, sm, X, b, reg_corner, x0)

    return x_reg, res, eta, reg_corner