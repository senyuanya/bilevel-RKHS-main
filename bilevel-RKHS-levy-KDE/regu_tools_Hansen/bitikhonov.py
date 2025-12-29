import numpy as np

def tikhonov(U, s, V, b, lambdas, x_0=None):
    """
    TIKHONOV Tikhonov regularization.

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

    # The overdetermined or square general-form case.
    gamma2 = (s[:, 0] / s[:, 1]) ** 2
    omega = np.linalg.solve(V, x_0)[:p]
    x0 = np.zeros(n)
    for i in range(ll):
        xi = (zeta + lambdas[i] ** 2 * (s[:, 1] ** 2) * omega) / (s[:, 0] ** 2 + lambdas[i] ** 2 * s[:, 1] ** 2)
        x_lambda[:, i] = V[:, :p] @ xi + x0
        rho[i] = lambdas[i] ** 2 * np.linalg.norm((beta - s[:, 0] * omega) / (gamma2 + lambdas[i] ** 2))
        eta[i] = np.linalg.norm(s[:, 1] * xi)


    # Calculate the product U[:, p:n].T @ b
    second_part = U[:, p:n].T @ b

    # Concatenate beta with second_part
    concatenated_vector = np.concatenate((beta, second_part))
    
    # Calculate U[:, :n] @ concatenated_vector
    product = U[:, :n] @ concatenated_vector
    
    # Final calculation of rho
    rho = np.sqrt(rho**2 + np.linalg.norm(b - product)**2)

    return x_lambda, rho, eta
