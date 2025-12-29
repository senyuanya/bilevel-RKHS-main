import numpy as np


def gsvdour(A, B):
    m, p = A.shape
    n, pb = B.shape
    if pb != p:
        raise ValueError("Matrix column mismatch between A and B")
    QA, A = np.linalg.qr(A,'reduced')
    m = p
    QB, B = np.linalg.qr(B,'reduced')
    
    Q, R = np.linalg.qr(np.concatenate((A, B),axis=0),'reduced')
    
    U, V, Z, C, S = csd(Q[:m, :], Q[m:m+n, :])
    X = R.T @ Z
    U = QA @ U
    V = QB @ V
    
    return U, V, X, C, S
    
def csd(Q1, Q2):
    m, p = Q1.shape
    n = Q2.shape[0]
    U, C, Z = np.linalg.svd(Q1)
    C = np.diag(C)
    Z = Z.T
    q = min(m, p)
    i = np.arange(q)
    j = np.arange(q-1, -1, -1)
    C[i, i] = C[j, j]
    U[:, i] = U[:, j]
    Z[:, i] = Z[:, j]
    S = Q2 @ Z
    
    k = np.max(np.concatenate(([0], 1+np.where(np.diag(C) <= 1/np.sqrt(2))[0])))
    V, _ = np.linalg.qr(S[:,:k],'complete')
    S = V.T @ S
    r = min(k, m)
    S[:, :r] = diagf(S[:, :r])
    if k < min(n, p):
        r = min(n, p)
        i = np.arange(k, n)
        j = np.arange(k, r)
        UT, ST, VT = np.linalg.svd(S[np.ix_(i, j)])
        ST = np.diag(ST)
        if k > 0:
            S[:k, j] = 0
        S[np.ix_(i,j)] = ST
        C[:, j] = C[:, j] @ VT.T
        V[:, i] = V[:, i] @ UT
        Z[:, j] = Z[:, j] @ VT.T
        i = np.arange(k, q)
        Q, R = np.linalg.qr(C[np.ix_(i, j)])
        C[np.ix_(i, j)] = diagf(R)
        U[:, i] = U[:, i] @ Q
    U, C = diagp(U, C, max(0, p - m))
    C = np.real(C)
    V, S = diagp(V, S, 0)
    S = np.real(S)
    return U, V, Z, C, S

def diagk(X, k):
    """
    Returns the k-th diagonal of X, even if X is a vector.
    """
    if not np.isscalar(X) and X.ndim > 1:  # Check if X is not a vector (i.e., it's a matrix)
        D = np.diag(X, k)
        D = D.reshape(-1)  # Ensure it's a column vector
    else:  # X is a vector
        if X.size > 0 and 0 <= k < X.shape[-1]:
            D = X[k]
        elif X.size > 0 and k < 0 and 1 - k <= X.shape[0]:
            D = X[-k]
        else:
            D = np.zeros(0, dtype=X.dtype)
    return D

def diagf(X):
    """
    Zeros all the elements off the main diagonal of X.
    """
    return np.triu(np.tril(X))

def diagp(Y, X, k):
    """
    Scales the columns of Y and the rows of X by unimodular factors to make the k-th diagonal of X real and positive.
    """
    # Extract the k-th diagonal of X
    D = np.diag(X, k)
    
    # Find indices where the real part is negative or the imaginary part is non-zero
    j = np.where((np.real(D) < 0) | (np.imag(D) != 0))[0]
    
    # Scale the diagonal elements to make them real and positive
    D = np.conj(D[j]) / np.abs(D[j])
    
    # Scale the corresponding columns of Y and rows of X
    Y[:, j] = Y[:, j] @ np.diag(D).T
    X[j, :] = np.diag(D) @ X[j, :]
    
    # Ensure that any potential negative zeros are set to positive zeros
    X = X + 0 
    return Y, X