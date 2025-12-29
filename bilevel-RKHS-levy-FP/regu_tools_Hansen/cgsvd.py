import numpy as np
from scipy.linalg import inv
# from pygsvd import pygsvd
# from gsvdm2py import gsvd
from regu_tools_Hansen.gsvd import gsvdour
def cgsvd(A, L):
    """
    Compact generalized SVD (GSVD) of a matrix pair in regularization problems.

    Parameters:
    A : array_like
        The first matrix in the matrix pair.
    L : array_like
        The second matrix in the matrix pair.

    Returns:
    U : ndarray
        The left singular vectors of A.
    sm : ndarray
        A 2-column array with the generalized singular values (sigma, mu).
    X : ndarray
        The right singular vectors of A.
    V : ndarray
        The left singular vectors of L.
    W : ndarray
        The inverse of X.
    """
    # Perform GSVD using pygsvd
    # U, V, W, C, S = gsvd(A, L)
    # print("矩阵 L 的秩:", np.linalg.matrix_rank(L))
    U, V, W, C, S = gsvdour(A, L)#mode="econ", eng=None) 
    # print("W:",W.shape)
    # C, S, W, U, V = pygsvd.gsvd(A, L, full_matrices = False, extras='uv')
    # Determine the dimensions
    m, n = A.shape
    p, _ = L.shape
    
    if m >= n:
        # The overdetermined or square case
        q = min(p, n)
        sm = np.column_stack((np.diag(C[:q, :q]), np.diag(S[:q, :q])))
        X = np.linalg.inv(W.T)
    else:
        # The underdetermined case
        sm = np.column_stack((np.diag(C[:m+p-n, n-m:p]), np.diag(S[n-m:p, n-m:p])))
    
    # Calculate W as the inverse transpose of X
    X = np.linalg.inv(W.T)
    X = X[:, n - m:n]
    W = W.T
    
    return U, sm, X, V, W

# def csvd(A, L=None):
#     """
#     扩展的奇异值分解 (SVD) 函数，可以执行标准 SVD 或广义奇异值分解 (GSVD)。
    
#     如果只提供 A，则计算 A 的标准奇异值分解（SVD）。
#     如果同时提供 A 和 L，则计算矩阵对 (A, L) 的广义奇异值分解（GSVD）。
    
#     参数
#     ----------
#     A : numpy ndarray
#         输入矩阵 (m x n)。
#     L : numpy ndarray, 可选
#         用于 GSVD 的第二个矩阵 (p x n)。如果为 None，则对 A 进行标准 SVD。
    
#     返回
#     -------
#     U : numpy ndarray
#         A 的左奇异向量（对于 SVD）或第一个矩阵的左奇异向量（对于 GSVD）。
#     sm : numpy ndarray
#         SVD 的奇异值或 GSVD 的广义奇异值（二维数组，包含 sigma 和 mu）。
#     X : numpy ndarray
#         A 的右奇异向量（对于 SVD）或共享的右奇异向量（对于 GSVD）。
#     V : numpy ndarray, 可选
#         第二个矩阵 L 的左奇异向量（仅适用于 GSVD）。
#     W : numpy ndarray, 可选
#         X 的逆（仅适用于 GSVD）。
#     """
#     if L is None:
#         # 标准 SVD 的计算
#         U, sig, Vt = np.linalg.svd(A, full_matrices=False)
#         return U, sig, Vt.T
#     else:
#         # 广义 SVD 的计算
#         # Step 1: 计算 A 和 L 的 QR 分解
#         QA, RA = np.linalg.qr(A)
#         QB, RB = np.linalg.qr(L)
        
#         # Step 2: 计算联合矩阵 [RA; RB] 的 SVD
#         S, sigma, X = np.linalg.svd(np.vstack([RA, RB]), full_matrices=False)
        
#         # Step 3: 计算广义奇异值
#         m, n = A.shape
#         p, _ = L.shape
        
#         if m >= n:
#             q = min(p, n)
#             sm = np.column_stack((sigma[:q], np.sqrt(1 - sigma[:q]**2)))
#         else:
#             sm = np.column_stack((sigma[:m+p-n], np.sqrt(1 - sigma[:m+p-n]**2)))
        
#         # Step 4: 计算 U 和 V 矩阵
#         U = QA @ S[:m, :]
#         V = QB @ S[m:, :]
        
#         # Step 5: 计算 W
#         W = np.linalg.inv(X.T)
#         X = X[:, :n]
        
#         return U, sm, X, V, W

# def cgsvd(A, L):
#     """
#     CGSVD Compact generalized SVD (GSVD) of a matrix pair in regularization problems.
    
#      sm = cgsvd(A,L)
#      [U,sm,X,V] = cgsvd(A,L) ,  sm = [sigma,mu]
#      [U,sm,X,V,W] = cgsvd(A,L) ,  sm = [sigma,mu]
    
#      Computes the generalized SVD of the matrix pair (A,L). The dimensions of
#      A and L must be such that [A;L] does not have fewer rows than columns.
    
#      If m >= n >= p then the GSVD has the form:
#         [ A ] = [ U  0 ]*[ diag(sigma)      0    ]*inv(X)
#         [ L ]   [ 0  V ] [      0       eye(n-p) ]
#                          [  diag(mu)        0    ]
#      where
#         U  is  m-by-n ,    sigma  is  p-by-1
#         V  is  p-by-p ,    mu     is  p-by-1
#         X  is  n-by-n .
    
#      Otherwise the GSVD has a more complicated form (see manual for details).
    
#      A possible fifth output argument returns W = inv(X).
     
#      Reference: C. F. Van Loan, "Computing the CS and the generalized 
#      singular value decomposition", Numer. Math. 46 (1985), 479-491. 
     
#      Per Christian Hansen, DTU Compute, August 22, 2009. 

#     """
#     # Ensure the inputs are full (dense) matrices
#     A = np.atleast_2d(A)
#     L = np.atleast_2d(L)

#     # Initialization
#     m, n = A.shape
#     p, n1 = L.shape

#     if n1 != n:
#         raise ValueError('Number of columns in A and L must be the same')
#     if m + p < n:
#         raise ValueError('Dimensions must satisfy m+p >= n')

#     # Perform the GSVD using scipy
#     # U, V, W, C, S = gsvd(A, L)
#     U, V, C, S, W =  pygsvd(A, L)

#     if m >= n:
#         # The overdetermined or square case.
#         q = min(p, n)
#         sm = np.hstack([np.diag(C[:q]), np.diag(S[:q])])
#         if cgsvd.__code__.co_argcount < 3:  # Simulating nargout < 2
#             return sm
#         else:
#             # Full decomposition
#             X = np.linalg.inv(W.T)
#     else:
#         # The underdetermined case.
#         sm = np.hstack([np.diag(C[:m + p - n, n - m:]), np.diag(S[n - m:n])])
#         if cgsvd.__code__.co_argcount < 3:  # Simulating nargout < 2
#             return sm
#         else:
#             # Full decomposition
#             X = np.linalg.inv(W.T)
#             X = X[:, n - m:n]

#     if cgsvd.__code__.co_argcount == 5:  # Simulating nargout == 5
#         return U, sm, X, V, W.T
#     return U, sm, X, V, W