import numpy as np
from iDARR.Lcurve import Lcurve

def cg_rkhs(Sigma_D, f, K, stop_rule, nsr, NoStop):
    """
    For the coordinate form RKHS regularization 
        min_c ||\Sigma c -f||_2^2 + lambda ||c||_{\sigma}^2,
     solve it by iterative regularization method cojugate gradient (CG).
   
    Inputs:
      Sigma_D, f: the loss function ||Sigma_D c- f||_2^2
      K: maximum iteration
      stop_rule:
          'DP': discrepancy principle
          'LC': L-curve
      nsr: noise norm. Set to 0 if unkown
      NoStop: Stop iteration if early stopping rule is satisfied.
   
    Outputs:
      X: store the first K regularized solutions
      res: strore residual norm of the first K regularized solution
      eta: strore solution norm of the first K regularized solution
      iter_stop: the early seopping iteration estimated by DP.

    """

    m, n = Sigma_D.shape
    if m != n or n != len(f):
        raise ValueError('Inconsistent dimensions')

    if stop_rule == 'LC':
        # nsr = 0  
        NoStop = 'on'  # no noise level provided--the iteration should run to complete and then use L-curve 
    elif stop_rule == 'DP':
        if nsr == 0:
            raise ValueError('Need noise norm for Discrepancy Principle')
        else:
            DP_tol = 1.001 * nsr  # 1.001*noise norm

    iter_stop = 0  # initial stopping iteration
    flag = False  # indicate whether we have found an early stopping iteration 

    # prepare for CG iteration
    X = np.zeros((n, K))
    res = np.zeros(K)
    eta = np.zeros(K)

    c = np.zeros(n)
    p = f
    v0 = f
    v_old = v0
   
    for k in range(K):    
        q = Sigma_D @ p
        alpha = (v_old.T @ Sigma_D @ v_old) / (q.T @ q)
        c += alpha * p
        v_new = v_old - alpha * Sigma_D @ p
        beta = (v_new.T @ Sigma_D @ v_new) / (v_old.T @ Sigma_D @ v_old)
        p = v_new + beta * p
        v_old = v_new

        X[:, k] = c
        res[k] = np.linalg.norm(Sigma_D @ c - f)
        eta[k] = np.sqrt(c.T @ Sigma_D @ c)

        # Determine early stopping iteration by DP
        if stop_rule == 'DP' and not flag and abs(res[k]) <= DP_tol:
            iter_stop = k
            flag = True

        if NoStop == 'off' and flag:
            X = X[:, :iter_stop]
            res = res[:iter_stop]
            eta = eta[:iter_stop]
            print(f'[Early stop: DP is satisfied], k_DP={iter_stop}')
            break

    # Estimate early stopping iteration by L-curve
    if stop_rule == 'LC':
        iter_stop, _ = Lcurve(res, eta, 0)
        print(f'[LC is satisfied], k_LC={iter_stop}')

    return X, res, eta, iter_stop

