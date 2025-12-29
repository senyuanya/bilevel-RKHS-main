import numpy as np

def filter_singular_values(B, tol=1e-5):
    """
    通过奇异值分解（SVD）筛选矩阵 B 的奇异值，去除小于 tol 的奇异值。
    
    参数:
    - B: 要筛选的矩阵
    - tol: 筛选阈值，可以是绝对值或相对值（基于最大奇异值）
    
    返回:
    - B_filtered: 筛选后的矩阵
    """
    U, s, Vt = np.linalg.svd(B, full_matrices=False)
    # 计算相对阈值（可选）
    # tol = tol * np.max(s)
    # 筛选奇异值
    s_filtered = np.where(s < tol, tol, s)
    # 重构筛选后的 B
    B_filtered = U @ np.diag(s_filtered) @ Vt
    return B_filtered


def filter_eigenvalues(B, tol=1e-5):
    """
    对对称方阵 B 进行特征值分解，并筛选小于 tol 的特征值。
    
    参数:
    - B: 要筛选的方阵（必须是对称的）。
    - tol: 筛选阈值，可以是绝对值或相对值（基于最大特征值）。
    
    返回:
    - B_filtered: 筛选后的矩阵。
    """
    # 特征值分解
    eigenvalues, Q = np.linalg.eigh(B)
    
    # 筛选特征值
    eigenvalues_filtered = np.where(eigenvalues < tol, tol, eigenvalues)

    # 重构筛选后的 B
    B_filtered = Q @ np.diag(eigenvalues_filtered) @ Q.T
    return B_filtered