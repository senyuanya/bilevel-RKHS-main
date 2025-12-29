import numpy as np
import matplotlib.pyplot as plt

def plot_lc(rho, eta, marker='-', ps=1, reg_param=None):
    """
    PLOT_LC Plot the L-curve.
    
     plot_lc(rho,eta,marker,ps,reg_param)
    
     Plots the L-shaped curve of the solution norm
        eta = || x ||      if   ps = 1
        eta = || L x ||    if   ps = 2
     as a function of the residual norm rho = || A x - b ||.  If ps is
     not specified, the value ps = 1 is assumed.
    
     The text string marker is used as marker.  If marker is not
     specified, the marker '-' is used.
    
     If a fifth argument reg_param is present, holding the regularization
     parameters corresponding to rho and eta, then some points on the
     L-curve are identified by their corresponding parameter.
    
     Per Christian Hansen, IMM, 12/29/97.
    """
    # Set defaults
    np_points = 10  # Number of identified points
    n = len(rho)
    ni = round(n / np_points)

    # Make plot
    plt.figure()
    plt.loglog(rho[1:-1], eta[1:-1])
    ax = plt.axis()
    
    # 检查 eta 和 rho 的范围
    if (np.max(eta) / np.min(eta) > 10) or (np.max(rho) / np.min(rho) > 10):
        # 假设 nargin >= 5
        plt.loglog(rho, eta, marker)
        plt.loglog(rho[ni-1:n:ni], eta[ni-1:n:ni], 'x')
        plt.axis(ax)
        
        for k in range(ni-1, n, ni):
            plt.text(rho[k], eta[k], str(reg_param[k]))

    else:

        plt.plot(rho, eta, marker)
        plt.plot(rho[ni-1:n:ni], eta[ni-1:n:ni], 'x')
        plt.axis(ax)
        
        for k in range(ni-1, n, ni):
            plt.text(rho[k], eta[k], str(reg_param[k]))
    

    
    # 添加标签和标题
    plt.xlabel('residual norm || A x - b ||_2')
    if ps == 1:
        plt.ylabel('solution norm || x ||_2')
    else:
        plt.ylabel('solution semi-norm || L x ||_2')
    plt.title('L-curve')
    
    # 显示图表
    plt.show()
