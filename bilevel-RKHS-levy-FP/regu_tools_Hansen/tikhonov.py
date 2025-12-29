import numpy as np

def tikhonov(U, s, V, b, lambdas, x_0=None):
    """
    TIKHONOV Tikhonov regularization.
    
     [x_lambda,rho,eta] = tikhonov(U,s,V,b,lambda,x_0)
     [x_lambda,rho,eta] = tikhonov(U,sm,X,b,lambda,x_0) ,  sm = [sigma,mu]
    
     Computes the Tikhonov regularized solution x_lambda, given the SVD or
     GSVD as computed via csvd or cgsvd, respectively.  If the SVD is used,
     i.e. if U, s, and V are specified, then standard-form regularization
     is applied:
        min { || A x - b ||^2 + lambda^2 || x - x_0 ||^2 } .
     If, on the other hand, the GSVD is used, i.e. if U, sm, and X are
     specified, then general-form regularization is applied:
        min { || A x - b ||^2 + lambda^2 || L (x - x_0) ||^2 } .
    
     If an initial estimate x_0 is not specified, then x_0 = 0 is used.
    
     Note that x_0 cannot be used if A is underdetermined and L ~= I.
    
     If lambda is a vector, then x_lambda is a matrix such that
        x_lambda = [ x_lambda(1), x_lambda(2), ... ] .
    
     The solution norm (standard-form case) or seminorm (general-form
     case) and the residual norm are returned in eta and rho.
    
     Per Christian Hansen, DTU Compute, April 14, 2003.
    
     Reference: A. N. Tikhonov & V. Y. Arsenin, "Solutions of Ill-Posed
     Problems", Wiley, 1977.

    """
    lambdas= [lambdas]
    if np.min(lambdas) < 0:
        raise ValueError('Illegal regularization parameter lambda')

    m = U.shape[0]
    n = V.shape[0]
    p, ps = s.shape 
    beta = U[:, :p].T @ b
    zeta = s[:, 0] * beta
    ll = len(lambdas)
    x_lambda = np.zeros((n, ll))
    rho = np.zeros(ll)
    eta = np.zeros(ll)

    # Treat each lambda separately.
    if ps == 1:
        omega = V.T @ x_0
        # print("omega:",omega)
        # print("s:",s)
        # print("s:",zeta)
        
        for i in range(ll):
            # print("lambdas[i]:",lambdas[i].shape)
            # print("(s ** 2 + lambdas[i] ** 2):",(s ** 2 + lambdas[i] ** 2).shape)
            # print("((zeta + lambdas[i] ** 2 * omega) :",(zeta + lambdas[i] ** 2 * omega) .shape)
            # print("V[:, :p] @ ((zeta + lambdas[i] ** 2 * omega) / (s ** 2 + lambdas[i] ** 2)[:,0]):",(V[:, :p] @ ((zeta + lambdas[i] ** 2 * omega) / (s ** 2 + lambdas[i] ** 2)[:,0])) .shape)
            x_lambda[:, i] = V[:, :p] @ ((zeta + lambdas[i] ** 2 * omega) / (s ** 2 + lambdas[i] ** 2)[:,0])
            rho[i] = (lambdas[i] ** 2) * np.linalg.norm((beta - s * omega) / (s ** 2 + lambdas[i] ** 2))
            eta[i] = np.linalg.norm(x_lambda[:, i])
        if len(U) > p:
            rho = np.sqrt(rho ** 2 + np.linalg.norm(b - U[:, :n] @ np.concatenate([beta, U[:, p:].T @ b])) ** 2)

    elif m >= n:
        # The overdetermined or square general-form case.
        gamma2 = (s[:, 0] / s[:, 1]) ** 2
        omega = np.linalg.solve(V, x_0)[:p]
        if p == n:
            x0 = np.zeros(n)
        else:
            x0 = V[:, p:n] @ U[:, p:n].T @ b
        for i in range(ll):
            xi = (zeta + lambdas[i] ** 2 * (s[:, 1] ** 2) * omega) / (s[:, 0] ** 2 + lambdas[i] ** 2 * s[:, 1] ** 2)
            x_lambda[:, i] = V[:, :p] @ xi + x0
            rho[i] = lambdas[i] ** 2 * np.linalg.norm((beta - s[:, 0] * omega) / (gamma2 + lambdas[i] ** 2))
            eta[i] = np.linalg.norm(s[:, 1] * xi)
        if len(U) > p:
            # Check dimensions of U, b, beta
            # print(f"U shape: {U.shape}")
            # print(f"b shape: {b.shape}")
            # print(f"beta shape: {beta.shape}")
            
            # Calculate the product U[:, p:n].T @ b
            second_part = U[:, p:n].T @ b
            # print(f"second_part shape: {second_part.shape}")
            
            # Concatenate beta with second_part
            concatenated_vector = np.concatenate((beta, second_part))
            # print(f"concatenated_vector shape: {concatenated_vector.shape}")
            
            # Calculate U[:, :n] @ concatenated_vector
            product = U[:, :n] @ concatenated_vector
            # print(f"product shape: {product.shape}")
            
            # Final calculation of rho
            rho = np.sqrt(rho**2 + np.linalg.norm(b - product)**2)
            # print(f"Final rho shape: {rho.shape}")
            # rho = np.sqrt(rho ** 2 + np.linalg.norm(b - U[:, :n] @ np.concatenate([beta, U[:, p:].T @ b])) ** 2)

    else:
        # The underdetermined general-form case.
        gamma2 = (s[:, 0] / s[:, 1]) ** 2
        if x_0 is not None:
            raise ValueError('x_0 not allowed in this case')
        if p == m:
            x0 = np.zeros(n)
        else:
            x0 = V[:, p:m] @ U[:, p:m].T @ b
        for i in range(ll):
            xi = zeta / (s[:, 0] ** 2 + lambdas[i] ** 2 * s[:, 1] ** 2)
            x_lambda[:, i] = V[:, :p] @ xi + x0
            rho[i] = lambdas[i] ** 2 * np.linalg.norm(beta / (gamma2 + lambdas[i] ** 2))
            eta[i] = np.linalg.norm(s[:, 1] * xi)

    return x_lambda, rho, eta
