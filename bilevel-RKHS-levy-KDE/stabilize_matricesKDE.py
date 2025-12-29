import numpy as np
from scipy.linalg import sqrtm
import matplotlib.pyplot as plt

def stabilize_matricesKDE(A, A_ts, B, A_tol = 1e-12, B_tol = 1e-12, plot_figures = True):
    """
    Stabilize input matrices A and B for CGSVD by:
    - Truncating small singular values of A (TSVD)
    - Regularizing B via spectral lifting and computing L = sqrt(B_reg)
    """
    # SVD for A
    UA, sA, VhA = np.linalg.svd(A, full_matrices=False)
    A_mask = sA > A_tol
    A_stable = UA[:, A_mask] @ np.diag(sA[A_mask]) @ VhA[A_mask, :]
    cond_A_trunc = sA[A_mask].max() / sA[A_mask].min()

    # SVD for A_ts
    UA_ts, sA_ts, VhA_ts = np.linalg.svd(A_ts, full_matrices=False)
    A_mask_ts = sA_ts > A_tol
    A_stable_ts = UA_ts[:, A_mask_ts] @ np.diag(sA_ts[A_mask_ts]) @ VhA_ts[A_mask_ts, :]
    cond_A_ts_trunc = sA_ts[A_mask_ts].max() / sA_ts[A_mask_ts].min()

    # Eigendecomposition for B
    eigvals, eigvecs = np.linalg.eigh(B)
    B_lifted = np.clip(eigvals, B_tol, None)
    B_reg = eigvecs @ np.diag(B_lifted) @ eigvecs.T
    cond_B_reg = B_lifted.max() / B_lifted.min()

    if plot_figures:
        # Spectrum plot for A
        plt.figure(figsize=(6, 4))
        plt.semilogy(sA, label='Singular values of A')
        plt.axhline(y=A_tol, color='r', linestyle='--', label='A_tol threshold')
        plt.title("SVD Spectrum of A")
        plt.xlabel("Index")
        plt.ylabel("Singular value (log scale)")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.show()
        
        # Spectrum plot for A_ts
        plt.figure(figsize=(6, 4))
        plt.semilogy(sA_ts, label='Singular values of A_ts')
        plt.axhline(y=A_tol, color='r', linestyle='--', label='A_tol threshold')
        plt.title("SVD Spectrum of A_ts")
        plt.xlabel("Index")
        plt.ylabel("Singular value (log scale)")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.show()
        
        # Spectrum plot for B
        plt.figure(figsize=(6, 4))
        plt.semilogy(np.sort(eigvals)[::-1], label='Eigenvalues of B')
        plt.axhline(y=B_tol, color='r', linestyle='--', label='B_tol threshold')
        plt.title("Eigenvalue Spectrum of B")
        plt.xlabel("Index")
        plt.ylabel("Eigenvalue (log scale)")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.show()

    info = {
        'original_A_cond': np.linalg.cond(A),
        'truncated_A_cond': cond_A_trunc,
        'original_A_ts_cond': np.linalg.cond(A_ts),
        'truncated_A_ts_cond': cond_A_ts_trunc,
        'original_B_cond': np.linalg.cond(B),
        'regularized_B_cond': cond_B_reg,
        'A_rank_retained': A_mask.sum(),
        'B_eig_retained': (eigvals > B_tol).sum()
    }

    return A_stable, A_stable_ts, B_reg, info
