import numpy as np
from scipy.linalg import sqrtm
from regu_tools_Hansen.cgsvd import cgsvd
from regu_tools_Hansen.tikhonov import tikhonov


def Tikh_auto_basis_SGD_nesterov_norm(A, b, A_ts, b_ts, B, scale_value, rho_val, method):
    """
    Use SGD combined with Nesterov momentum and gradient normalization to automatically select the Tikhonov regularization parameter.
    
    After normalization, the update rule is:

    γ_{t+1} = γ_t – α_t * (g_t / ‖g_t‖)

    where g_t is the outer gradient dF/dγ computed by the chain rule.
    """

    # 计算矩阵 B 的平方根，并执行 CGSVD 分解
    L = np.real(sqrtm(B))
    U, sm, X, _, _ = cgsvd(A, L)
    
    # 初始化 x0
    n = A.shape[1]
    x0 = np.zeros(n)
    
    # 初始化 gamma（对应正则化参数 lambda = 10^γ）
    gamma = 0.0
    num_iterations = 5000
    lr0 = 0.6# Compound:sinx: L2 l2 lr0 = 0.06; momentum_factor = 0.9; -0.5x L2 l2 lr0 = 0.06; momentum_factor = 0.99;  Laplace:sinx: L2 l2 lr0 = 0.06; momentum_factor = 0.9; -0.5x L2 l2 lr0 = 0.5;lr0 = 0.05; momentum_factor = 0.9
    gamma_tol=1e-4
    loss_tol=1e-6
    earlystop_window=500
    # Nesterov 动量参数
    momentum_factor = 0.99#9# Compoundlevy np.sin(x): lr0 = 0.005; momentum_factor = 0.95; num_iterations = 10000;  -0.5*x: lr0 = 0.004; momentum_factor = 0.99; num_iterations = 5000
                           # Laplacejump  np.sin(x): lr0 = 0.005; momentum_factor = 0.95; num_iterations = 10000; -0.5*x: lr0 = 0.009; momentum_factor = 0.99; num_iterations = 5000
    v_gamma = 0.0  # 初始化动量

    eps = 1e-12       # 防止除零
    threshold = 1e-6 # 新增阈值，判断梯度是否“过小”
    
    # 从 CGSVD 中提取 sigma 和 mu（用于构造 Hessian 近似）
    sigma = sm[:, 0]
    mu = sm[:, 1]
    
    # 初始化记录列表
    loss_history = []
    gamma_history = []
    iteration_history = []
    xreg7_history = []
    # Early-stopping counter
    stable_iter = 0
    
    for it in range(1, num_iterations + 1):
        # 使用动态学习率： lr_t = lr0 / sqrt(it)
        lr_t = lr0 / np.sqrt(it)

        # Nesterov 预更新：计算 lookahead 位置的 gamma
        gamma_lookahead = gamma - momentum_factor * v_gamma
        lambd_lookahead = 10 ** gamma_lookahead
        
    
        # 内层求解：利用 lookahead 位置的 lambda 求解 Tikhonov 正则化问题
        x_reg, res, eta = tikhonov(U, sm, X, b, lambd_lookahead, x0)
        W_rkhs = x_reg.reshape(-1)
        
        ### 梯度计算部分 ###
        # 计算 D = sigma^2 + lambda * mu^2，其中 lambda 采用 lookahead 值
        D = sigma**2 + lambd_lookahead * mu**2
        tol_val = 1e-12
        D_inv = np.array([1/d if d > tol_val else 0 for d in D])
        D_inv_matrix = np.diag(D_inv)
        
        # 计算近似 Hessian 的逆：dL/dW 关于 W 的导数
        dL_dWW = X @ D_inv_matrix @ X.T
        
        # 计算 dL/dW 关于 gamma 的导数：
        # dL/dWgamma = log(10) * lambda * (B * W)
        dL_dWgamma = np.log(10) * lambd_lookahead * np.dot(B, W_rkhs)
        
        # 链式法则：计算 W 关于 gamma 的梯度
        dW_dgamma = - dL_dWW @ dL_dWgamma
        
        # 计算验证数据上的残差，并构造上层损失梯度
        residual = A_ts.T @ (A_ts @ W_rkhs - b_ts)
        dF_dgamma = 2 * residual.T @ dW_dgamma * scale_value
        
        norm_dF = np.linalg.norm(dF_dgamma) + 1e-12
        dF_dgamma_normalized = dF_dgamma / norm_dF
      
        # Nesterov 动量更新：先更新动量，再更新 gamma
        v_gamma = momentum_factor * v_gamma + lr_t * dF_dgamma_normalized
        gamma = gamma - v_gamma
        # print("v_gamma:", v_gamma)
        # 计算验证数据上的外层损失 ||A_ts * W - b_ts||^2
        diff_val = A_ts @ W_rkhs - b_ts
        outer_loss = np.linalg.norm(diff_val)**2
        loss_history.append(outer_loss)
        gamma_history.append(gamma)
        iteration_history.append(it)
        xreg7_history.append(x_reg)
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
            
        if it % 100 == 0:
            print(f"Iteration {it}, Gamma: {gamma:.4f}, Lambda: {10**gamma:.8f}, Loss: {outer_loss:.6f}")
    
    # 最后利用最优 gamma 计算对应的 lambda，并重新求解
    lambd = 10 ** gamma
    x_reg, res, eta = tikhonov(U, sm, X, b, lambd, x0)
    
    return x_reg, res, eta, lambd, loss_history, gamma_history, iteration_history, xreg7_history
