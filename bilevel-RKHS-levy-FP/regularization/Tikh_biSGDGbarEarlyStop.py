import numpy as np
from scipy.linalg import sqrtm
from regu_tools_Hansen.cgsvd import cgsvd
from regu_tools_Hansen.tikhonov import tikhonov

def Tikh_auto_basis_SGD_nesterov_normalized(A, b, A_ts, b_ts, B, scale_value, rho_val, method):
    """
    Use SGD combined with Nesterov momentum and gradient normalization to automatically select the Tikhonov regularization parameter.
    Early stopping: if gamma and loss both change by less than tol for `earlystop_window` consecutive iterations, stop.
    """

    # Compute sqrt(B) and CGSVD
    L = np.real(sqrtm(B))
    U, sm, X, _, _ = cgsvd(A, L)
    n = A.shape[1]
    x0 = np.zeros(n)

    # Initialize gamma and momentum
    gamma = 0.0
    v_gamma = 0.0

    num_iterations=5000
    lr0=0.005
    momentum_factor=0.95
    gamma_tol=1e-4
    loss_tol=1e-6
    earlystop_window=500

    eps = 1e-12  # avoid div0

    # Extract sigma and mu for Hessian approx
    sigma = sm[:, 0]
    mu = sm[:, 1]

    # Histories
    loss_history = []
    gamma_history = []
    iteration_history = []
    xreg7_history = []  # record x_reg history
    # Early-stopping counter
    stable_iter = 0

    for it in range(1, num_iterations + 1):
        lr_t = lr0 / np.sqrt(it)

        # Nesterov lookahead
        gamma_lookahead = gamma - momentum_factor * v_gamma
        lambd_lookahead = 10 ** gamma_lookahead

        # Inner Tikhonov solve
        x_reg, res, eta = tikhonov(U, sm, X, b, lambd_lookahead, x0)
        W_rkhs = x_reg.reshape(-1)

        # Hessian approx
        D = sigma**2 + lambd_lookahead * mu**2
        D_inv = np.array([1/d if d > 1e-12 else 0 for d in D])
        D_inv_matrix = np.diag(D_inv)

        # Grad w.r.t gamma
        dL_dWW = X @ D_inv_matrix @ X.T
        dL_dWgamma = np.log(10) * lambd_lookahead * (B @ W_rkhs)
        dW_dgamma = - dL_dWW @ dL_dWgamma

        # Outer gradient
        residual = A_ts.T @ (A_ts @ W_rkhs - b_ts)
        dF_dgamma = 2 * residual.T @ dW_dgamma * scale_value
        dF_dgamma_normalized = dF_dgamma / (np.linalg.norm(dF_dgamma) + eps)

        # Momentum update
        v_gamma = momentum_factor * v_gamma + lr_t * dF_dgamma_normalized
        gamma = gamma - v_gamma

        # Compute loss
        diff_val = A_ts @ W_rkhs - b_ts
        outer_loss = np.linalg.norm(diff_val)**2

        # Record
        gamma_history.append(gamma)
        loss_history.append(outer_loss)
        xreg7_history.append(x_reg)
        iteration_history.append(it)
        # Check stability
        if it > 1:
            dg = abs(gamma_history[-1] - gamma_history[-2])
            dl = abs(loss_history[-1] - loss_history[-2])
            if dg < gamma_tol and dl < loss_tol:
                stable_iter += 1
            else:
                stable_iter = 0

            if stable_iter >= earlystop_window:
                print(f"Early stopping at iteration {it}: gamma and loss stabilized for {earlystop_window} steps.")
                break

        # Optional logging
        if it % 100 == 0:
            print(f"Iteration {it}, Gamma: {gamma:.6f}, Lambda: {10**gamma:.6e}, Loss: {outer_loss:.6f}")

    # Final solve with best gamma
    lambd = 10 ** gamma
    x_reg, res, eta = tikhonov(U, sm, X, b, lambd, x0)

    return x_reg, res, eta, lambd, loss_history, gamma_history, iteration_history, xreg7_history
