import numpy as np

def normalize_byL2norm(u, f, dx):
    d = u.ndim
    n = np.linalg.norm(u, ord=2, axis=d-1) * np.sqrt(dx)
    u_normed = u / n[:, np.newaxis]
    f_normed = f / n[:, np.newaxis]
    return u_normed, f_normed

