import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize_scalar

def gcv(U, s, b, method='Tikh'):
    r"""
    Compute and plot the GCV function for regularization and find its minimizer.
    """
    # Set default parameters
    npoints = 200
    smin_ratio = 16 * np.finfo(float).eps

    m, n = U.shape
    # Determine the shape of s
    s = np.atleast_2d(s)
    p, ps = s.shape
    
    # Initialization.
    beta = U.T @ b
    beta2 = np.linalg.norm(b)**2 - np.linalg.norm(beta)**2

    # If s is given as a two-column matrix, adjust s and beta accordingly.
    if ps == 2:
        # Reverse the order and compute the ratio elementwise.
        s = (s[::-1, 0] / s[::-1, 1]).flatten()
        beta = beta[::-1]
    else:
        # Ensure s is a one-dimensional array.
        s = s.flatten()

    # For functions that return output, we compute the minimum.
    find_min = True

    # Method: Tikhonov regularization
    if method.lower().startswith('tikh'):
        # Precompute squared singular values for use in Tikhonov
        s2 = s**2
        # Build vector of regularization parameters.
        reg_param = np.zeros(npoints)
        # In MATLAB, indices: reg_param(npoints) = max( s(p), s(1)*smin_ratio )
        # Here s[0] is s(1) and s[-1] is s(p) assuming s is sorted in descending order.
        reg_param[-1] = max(s[-1], s[0]*smin_ratio)
        ratio = (s[0] / reg_param[-1]) ** (1/(npoints-1))
        for i in range(npoints-2, -1, -1):
            reg_param[i] = ratio * reg_param[i+1]
        # Now, reg_param[0] == s[0] and reg_param[-1] == s[-1].
        
        # Intrinsic residual delta0.
        delta0 = beta2 if (m > n and beta2 > 0) else 0

        # Compute the GCV function values
        G_vals = np.zeros(npoints)
        for i in range(npoints):
            G_vals[i] = gcvfun(reg_param[i], s2, beta[:p], delta0, m - n, dsvd=False)

        # Plot the GCV function in a loglog scale.
        plt.figure()
        plt.loglog(reg_param, G_vals, '-')
        plt.xlabel(r'$\lambda$')
        plt.ylabel(r'$G(\lambda)$')
        plt.title('GCV function')

        # Find the minimizer if requested.
        if find_min:
            min_index = np.argmin(G_vals)
            # Determine the bounds for minimization.
            # Note: In the MATLAB code, reg_param is in descending order so that:
            #   lower_bound = reg_param[min(min_index+1, npoints)]
            #   upper_bound = reg_param[max(min_index-1, 1)]
            # Adjusting for 0-indexing:
            lb = reg_param[min(min_index+1, npoints-1)]
            ub = reg_param[max(min_index-1, 0)]
            # Ensure lower bound is less than upper bound.
            low_bound, high_bound = min(lb, ub), max(lb, ub)
            res = minimize_scalar(
                lambda lam: gcvfun(lam, s2, beta[:p], delta0, m - n, dsvd=False),
                bounds=(low_bound, high_bound),
                method='bounded'
            )
            reg_min = res.x
            minG = gcvfun(reg_min, s2, beta[:p], delta0, m - n, dsvd=False)
            print("reg_min:",reg_min)
            # Mark the minimizer on the plot.
            plt.loglog(reg_min, minG, '*r')
            plt.loglog([reg_min, reg_min], [minG/1000, minG], ':r')
            plt.title(r'GCV function, minimum at $\lambda = {:.4g}$'.format(reg_min))
            plt.show()

        else:
            reg_min = None

    # Method: Truncated SVD (or GSVD)
    elif method.lower().startswith('tsvd') or method.lower().startswith('tgsv'):
        # Compute rho2 (an array of length p-1).
        rho2 = np.zeros(p-1)
        rho2[-1] = abs(beta[-1])**2
        if m > n and beta2 > 0:
            rho2[-1] += beta2
        # Loop backward in the equivalent of MATLAB indexing:
        # For MATLAB: for k = p-2:-1:1, then in python, indices 0 to p-2 (inclusive), reversed.
        for i in reversed(range(0, p-1)):
            # Only update if i is not the last index; for i = p-2 down to 0,
            # using the relation: rho2(i) = rho2(i+1) + abs(beta(i+1))^2.
            if i < p-1 - 1:
                rho2[i] = rho2[i+1] + abs(beta[i+1])**2
            elif i == p-2:
                rho2[i] = rho2[i+1] + abs(beta[i+1])**2

        # Compute the GCV function values:
        G_vals = np.zeros(p-1)
        # The denominator in the GCV function here is: (m - k + (n-p))^2 for k = 1,...,p-1.
        for k in range(1, p):
            denominator = (m - k + (n - p))**2
            G_vals[k-1] = rho2[k-1] / denominator

        # Here reg_param is just the index (k) vector.
        reg_param = np.arange(1, p)

        # Plot the GCV function (using a semilogy plot).
        plt.figure()
        plt.semilogy(reg_param, G_vals, 'o')
        plt.xlabel('k')
        plt.ylabel(r'$G(k)$')
        plt.title('GCV function')

        # Find the minimum and mark it.
        if find_min:
            min_index = np.argmin(G_vals)
            reg_min = reg_param[min_index]  # reg_min is the index k at the minimum.
            minG = G_vals[min_index]
            plt.semilogy(reg_min, minG, '*r')
            plt.semilogy([reg_min, reg_min], [minG/1000, minG], ':r')
            plt.title('GCV function, minimum at k = {}'.format(reg_min))
            plt.show()
        else:
            reg_min = None

    # Method: Damped SVD (or GSVD)
    elif method.lower().startswith('dsvd') or method.lower().startswith('dgsv'):
        # Build vector of regularization parameters similarly to the Tikhonov branch.
        reg_param = np.zeros(npoints)
        reg_param[-1] = max(s[-1], s[0]*smin_ratio)
        ratio = (s[0] / reg_param[-1]) ** (1/(npoints-1))
        for i in range(npoints-2, -1, -1):
            reg_param[i] = ratio * reg_param[i+1]

        delta0 = beta2 if (m > n and beta2 > 0) else 0

        # For dsvd, we use the function with the flag set to True.
        G_vals = np.zeros(npoints)
        for i in range(npoints):
            G_vals[i] = gcvfun(reg_param[i], s, beta[:p], delta0, m - n, dsvd=True)

        plt.figure()
        plt.loglog(reg_param, G_vals, ':')
        plt.xlabel(r'$\lambda$')
        plt.ylabel(r'$G(\lambda)$')
        plt.title('GCV function')

        if find_min:
            min_index = np.argmin(G_vals)
            lb = reg_param[min(min_index+1, npoints-1)]
            ub = reg_param[max(min_index-1, 0)]
            low_bound, high_bound = min(lb, ub), max(lb, ub)
            res = minimize_scalar(
                lambda lam: gcvfun(lam, s, beta[:p], delta0, m - n, dsvd=True),
                bounds=(low_bound, high_bound),
                method='bounded'
            )
            reg_min = res.x
            minG = gcvfun(reg_min, s, beta[:p], delta0, m - n, dsvd=True)
            plt.loglog(reg_min, minG, '*r')
            plt.loglog([reg_min, reg_min], [minG/1000, minG], ':r')
            plt.title(r'GCV function, minimum at $\lambda = {:.4g}$'.format(reg_min))
            plt.show()
        else:
            reg_min = None

    elif method.lower().startswith('mtsv') or method.lower().startswith('ttls'):
        raise ValueError('The MTSVD and TTLS methods are not supported')
    else:
        raise ValueError('Illegal method')

    return reg_min, G_vals, reg_param


def gcvfun(lam, s2, beta, delta0, mn, dsvd=False):
    if not dsvd:
        # Tikhonov: use f = \lambda^2/(s^2 + \lambda^2)
        f = (lam**2) / (s2 + lam**2)
    else:
        # Damped SVD: here s2 is actually the vector s (unsquared)
        f = lam / (s2 + lam)
    num = np.linalg.norm(f * beta)**2 + delta0
    den = (mn + np.sum(f))**2
    G = num / den
    return G