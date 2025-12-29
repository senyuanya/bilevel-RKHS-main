import numpy as np
from scipy import optimize
from scipy import linalg
import matplotlib.pyplot as plt
from regu_tools_Hansen.plot_lc import plot_lc
from scipy.ndimage import gaussian_filter1d

def l_curve(U, sm, b, method='Tikh'):
    """
    
     L_CURVE Plot the L-curve and find its "corner".
     
      [reg_corner,rho,eta,reg_param] =
                       l_curve(U,s,b,method)
                       l_curve(U,sm,b,method)  ,  sm = [sigma,mu]
                       l_curve(U,s,b,method,L,V)
     
      Plots the L-shaped curve of eta, the solution norm || x || or
      semi-norm || L x ||, as a function of rho, the residual norm
      || A x - b ||, for the following methods:
         method = 'Tikh'  : Tikhonov regularization   (solid line )
         method = 'tsvd'  : truncated SVD or GSVD     (o markers  )
         method = 'dsvd'  : damped SVD or GSVD        (dotted line)
         method = 'mtsvd' : modified TSVD             (x markers  )
      The corresponding reg. parameters are returned in reg_param.  If no
      method is specified then 'Tikh' is default.  For other methods use plot_lc.
     
      Note that 'Tikh', 'tsvd' and 'dsvd' require either U and s (standard-
      form regularization) computed by the function csvd, or U and sm (general-
      form regularization) computed by the function cgsvd, while 'mtvsd'
      requires U and s as well as L and V computed by the function csvd.
     
      If any output arguments are specified, then the corner of the L-curve
      is identified and the corresponding reg. parameter reg_corner is
      returned.  Use routine l_corner if an upper bound on eta is required.
    
      Reference: P. C. Hansen & D. P. O'Leary, "The use of the L-curve in
      the regularization of discrete ill-posed problems",  SIAM J. Sci.
      Comput. 14 (1993), pp. 1487-1503.
    
      Per Christian Hansen, DTU Compute, October 27, 2010.

    """
    # Set defaults
    npoints = 200  # Number of points on the L-curve for Tikh and dsvd
    smin_ratio = 16 * np.finfo(float).eps  # Smallest regularization parameter

    # Initialization
    m, n = U.shape

    p, ps = sm.shape
    locate = True  # Assuming that we want to locate the "corner"
    beta = U.T @ b
    beta2 = np.linalg.norm(b)**2 - np.linalg.norm(beta)**2
    
    if ps == 1:
        s = sm
        beta = beta[:p]
        beta = beta[:, np.newaxis] 
    else:
        s = sm[p-1::-1, 0] / sm[p-1::-1, 1]
        beta = beta[p-1::-1]
    
    xi = beta[:p] / s
    # print("beta[:p] .shape:",beta[:p] .shape)
    # print("s .shape:",s.shape)
    xi[np.isinf(xi)] = 0
    
    eta = np.zeros((npoints,1))
    rho = np.zeros((npoints,1))
    reg_param = np.zeros((npoints,1))
    s2 = s**2
    reg_param[-1] = max(s[p-1], s[0] * smin_ratio)
    ratio = (s[0] / reg_param[-1])**(1/(npoints-1))
    for i in range(npoints-2, -1, -1):
        reg_param[i] = ratio * reg_param[i+1]
    for i in range(npoints):
        f = s2 / (s2 + reg_param[i]**2)
        # print("xi:",xi.shape)
        # print("f:",f.shape)
        eta[i] = np.linalg.norm(f * xi)
        rho[i] = np.linalg.norm((1 - f) * beta[:p])

    if m > n and beta2 > 0:
        rho = np.sqrt(rho**2 + beta2)
    marker = '-'
    txt = 'Tikh.'
    
    # Locate the "corner" of the L-curve, if required.
    if locate:
        reg_corner, rho_c, eta_c = l_corner(rho, eta, reg_param, U, sm, b, method)
    # print("rho_c:",rho_c) 
    # print("eta_c:",eta_c) 
    # print("reg_corner:",reg_corner) 
    plot_lc(rho, eta, marker, ps, reg_param)
    # print("np.min(rho)/100:",np.min(rho)/100)
    # print("np.min(eta)/100:",np.min(eta)/100)
    # print("eta_c:",eta_c)
    # print("rho_c:",rho_c)
    
    if locate:
        # print('[np.min(rho)/100, rho_c]:',[np.min(rho)/100, rho_c])
        # print('[eta_c, eta_c]:',[eta_c, eta_c])
        plt.figure()
        plt.loglog(rho[1:-1], eta[1:-1])
        ax = plt.axis()
        # plt.loglog([np.min(rho)/100, rho_c], [eta_c, eta_c], ':r',
        #            [rho_c, rho_c], [np.min(eta)/100, eta_c], ':r')
        plt.loglog(rho_c, eta_c, 'x')
        plt.text(rho_c, eta_c, str(reg_corner))
        plt.loglog([np.min(rho)/100, rho_c], [eta_c, eta_c], ':r')
        plt.loglog([rho_c, rho_c], [np.min(eta)/100, eta_c], ':r')
        plt.title(f'L-curve, {txt} corner at {reg_corner}')
        plt.axis(ax)
        plt.show()
    return reg_corner, rho, eta, reg_param


def l_corner(rho,eta,reg_param,U,s, b, method): # rho, eta, reg_param=None, U=None, s=None, b=None, method='none'
    '''
    L_CORNER Locate the "corner" of the L-curve.
    
     [reg_c,rho_c,eta_c] =
            l_corner(rho,eta,reg_param)
            l_corner(rho,eta,reg_param,U,s,b,method,M)
            l_corner(rho,eta,reg_param,U,sm,b,method,M) ,  sm = [sigma,mu]
    
     Locates the "corner" of the L-curve in log-log scale.
    
     It is assumed that corresponding values of || A x - b ||, || L x ||,
     and the regularization parameter are stored in the arrays rho, eta,
     and reg_param, respectively (such as the output from routine l_curve).
    
     If nargin = 3, then no particular method is assumed, and if
     nargin = 2 then it is issumed that reg_param = 1:length(rho).
    
     If nargin >= 6, then the following methods are allowed:
        method = 'Tikh'  : Tikhonov regularization
        method = 'tsvd'  : truncated SVD or GSVD
        method = 'dsvd'  : damped SVD or GSVD
        method = 'mtsvd' : modified TSVD,
     and if no method is specified, 'Tikh' is default.  If the Spline Toolbox
     is not available, then only 'Tikh' and 'dsvd' can be used.
    
     An eighth argument M specifies an upper bound for eta, below which
     the corner should be found.
    
     Per Christian Hansen, DTU Compute, January 31, 2015.
    '''
    # Ensure that rho and eta are column vectors.
    rho = np.atleast_1d(rho).flatten()
    eta = np.atleast_1d(eta).flatten()

    # Set threshold for skipping very small singular values in the
    # analysis of a discrete L-curve.
    s_thr = np.finfo(float).eps # Neglect singular values less than s_thr.
    # Set default parameters for treatment of discrete L-curve.
    deg   = 2  # Degree of local smooting polynomial.
    q     = 2  # Half-width of local smoothing interval.
    order = 4  # Order of fitting 2-D spline curve.
    # Initialization.
    if (len(rho) < order):
        print('Too few data points for L-curve analysis')
    p, ps = s.shape
    m, n = U.shape
    beta = U.T @ b
    b0 = b - np.dot(U, beta)
    if ps == 2:
        s = s[p-1::-1, 0] / s[p-1::-1, 1]
        beta = beta[p-1::-1]
    else:
        beta = beta[:, np.newaxis] 
    xi = beta / s
    if m > n:
        beta = np.append(beta, np.linalg.norm(b0))
        
    g = lcfun(reg_param, s, beta, xi)
    
    # Locate the corner.  If the curvature is negative everywhere,
    # then define the leftmost point of the L-curve as the corner.
    gi = np.argmin(g)
    x1 = reg_param[int(np.amin([gi, len(g)]))]
    x2 = reg_param[int(np.amax([gi, 0]))]
    reg_c = optimize.fminbound(lcfun, x1, x2, args = (s, beta, xi), full_output=False, disp=False)
    
    # reg_c = optimize.fminbound(lcfun, reg_param[max(gi-1, 0)], reg_param[min(gi+1, len(g)-1)],
    #                   args=(s, beta, xi), disp=False)  # lcfun = curvature
    
    # print("reg_c:",reg_c)
    kappa_max = -lcfun(reg_c, s, beta, xi)  # kappa_max = - curvature(reg_c, sig, beta, xi) # Maximum curvature.
    
    # if (nargout > 0), locate = 1; else locate = 0; end
    # print("reg_param:",reg_param.shape)
    # print("lr:",len(rho))
    if kappa_max < 0:
        lr = len(rho)-1
        reg_c = reg_param[lr]
        rho_c = rho[lr]
        eta_c = eta[lr]
        # print("reg_param[lr]:",reg_param[len(rho)])
    else:
        f = (s**2) / (s**2 + reg_c**2)
        eta_c = np.linalg.norm(f * xi)
        rho_c = np.linalg.norm((1-f) * beta[0:len(f)])
        if m > n:
            rho_c = np.sqrt(rho_c ** 2 + np.linalg.norm(b0)**2)
    return reg_c, rho_c, eta_c

def lcfun(lambd, sig, beta, xi):
    '''
    computes the NEGATIVE of the curvature. Adapted from Per Christian Hansen, DTU Compute, October 27, 2010.
    '''
    # Initialization.
    phi = np.zeros(lambd.shape)
    dphi = np.zeros(lambd.shape)
    psi = np.zeros(lambd.shape)
    dpsi = np.zeros(lambd.shape)
    eta = np.zeros(lambd.shape)
    rho = np.zeros(lambd.shape)
    if len(beta) > len(sig): # A possible least squares residual.
        LS = True
        rhoLS2 = beta[-1] ** 2
        beta = beta[:-1]
    else:
        LS = False
    # Compute some intermediate quantities.
    for jl, lam in enumerate(lambd):
        f = (sig**2) / (sig**2 + lam**2)
        cf = 1 - f # ok
        eta[jl] = np.linalg.norm(f * xi) # ok
        rho[jl] = np.linalg.norm(cf * beta)
        f1 = -2 * f * cf / lam 
        f2 = -f1 * (3 - 4*f)/lam
        phi[jl]  = np.sum(f*f1*np.abs(xi)**2) #ok
        psi[jl] = np.sum(cf*f1*np.abs(beta)**2)
        dphi[jl] = np.sum((f1**2 + f*f2)*np.abs(xi)**2)
        dpsi[jl] = np.sum((-f1**2 + cf*f2)*np.abs(beta)**2) #ok

    if LS: # Take care of a possible least squares residual.
        rho = np.sqrt(rho ** 2 + rhoLS2)

    # Now compute the first and second derivatives of eta and rho
    # with respect to lambda;
    deta  =  np.divide(phi, eta) #ok
    drho  = -np.divide(psi, rho)
    ddeta =  np.divide(dphi, eta) - deta * np.divide(deta, eta)
    ddrho = -np.divide(dpsi, rho) - drho * np.divide(drho, rho)

    # Convert to derivatives of log(eta) and log(rho).
    dlogeta  = np.divide(deta, eta)
    dlogrho  = np.divide(drho, rho)
    ddlogeta = np.divide(ddeta, eta) - (dlogeta)**2
    ddlogrho = np.divide(ddrho, rho) - (dlogrho)**2
    # curvature.
    curv = - np.divide((dlogrho * ddlogeta - ddlogrho * dlogeta),
        (dlogrho**2 + dlogeta**2)**(1.5))
    return curv

def csvd(A):
    '''
    computes the svd based on the size of A.
    Input:
        A is of Nm x Nu, where Nm are the number of measurements and Nu the number of unknowns
    Adapted from Per Christian Hansen, DTU Compute, October 27, 2010.
    '''
    Nm, Nu = A.shape
    if Nm >= Nu: # more measurements than unknowns
        u, sig, v = linalg.svd(A, full_matrices=False)
    else:
        v, sig, u = linalg.svd(A.T, full_matrices=False)
    return u, sig, v