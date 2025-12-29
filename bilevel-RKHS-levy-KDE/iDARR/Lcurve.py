import numpy as np
import matplotlib.pyplot as plt
from scipy.sparse import diags

def Lcurve(rho, eta, fig=0, rkhs_type='auto'):
    """
    
    CORNER Find corner of discrete L-curve via adaptive pruning algorithm.
    
    [k_corner,info] = corner(rho,eta,fig)
    
    Returns the integer k_corner such that the corner of the log-log
    L-curve is located at ( log(rho(k_corner)) , log(eta(k_corner)) ).
    
    The vectors rho and eta must contain corresponding values of the
    residual norm || A x - b || and the solution's (semi)norm || x ||
    or || L x || for a sequence of regularized solutions, ordered such
    that rho and eta are monotonic and such that the amount of
    regularization decreases as k increases.
    
    The second output argument describes possible warnings.
    Any combination of zeros and ones is possible.
      info = 000 : No warnings - rho and eta describe a discrete 
                      L-curve with a corner.
      info = 001 : Bad data - some elements of rho and/or eta are
                      Inf, NaN, or zero.
      info = 010 : Lack of monotonicity - rho and/or eta are not
                      strictly monotonic.
      info = 100 : Lack of convexity - the L-curve described by rho
                      and eta is concave and has no corner.
                      
    The warnings described above will also result in text warnings on the
    command line. Type 'warning off Corner:warnings' to disable all
    command line warnings from this function.
    
    If a third input argument is present, then a figure will show the discrete
    L-curve in log-log scale and also indicate the found corner.
    
    Reference: P. C. Hansen, T. K. Jensen and G. Rodriguez, "An adaptive
    pruning algorithm for the discrete L-curve criterion," J. Comp. Appl.
    Math., 198 (2007), 483-492.
      
    Per Christian Hansen and Toke Koldborg Jensen, DTU Compute, DTU;
    Giuseppe Rodriguez, University of Cagliari, Italy; Sept. 2, 2011.

    """
    # Initialization of data
    rho = np.array(rho).flatten()
    eta = np.array(eta).flatten()
    
    if len(rho) != len(eta):
        raise ValueError('Vectors rho and eta must have the same length')
    if len(rho) < 3:
        raise ValueError('Vectors rho and eta must have at least 3 elements')
    
    # Handle the 'fig' input
    if fig is None or fig < 0:
        fig = 0  # Default is no figure
    
    info = 0

    # Check for bad data (Inf, NaN, zeros)
    fin = np.isfinite(rho + eta)  # Check for NaN or Inf  NaN or Inf will cause trouble.
    nzr = (rho * eta) != 0        # Check for zeros A zero will cause trouble.
    kept = np.where(fin & nzr)[0]

    if len(kept) == 0:
        raise ValueError('Too many Inf/NaN/zeros found in data')
    if len(kept) < len(rho):
        info += 1
        print('Warning: Bad data - Inf, NaN or zeros found in data. Continuing with the remaining data')

    rho = rho[kept]  # rho and eta with bad data removed.
    eta = eta[kept]

    # Check for monotonicity
    if np.any(rho[:-1] < rho[1:]) or np.any(eta[:-1] > eta[1:]):
        info += 10
        print('Warning: Lack of monotonicity')

    # Prepare for adaptive algorithm
    nP = len(rho)  # Number of points.
    P = np.log10(np.column_stack((rho, eta)))  # Coordinates of the loglog L-curve.
    V = P[1:] - P[:-1]  # The vectors defined by these coordinates.
    v = np.sqrt(np.sum(V**2, axis=1))  # The length of the vectors.
    W = V / v[:, None]  # Normalized vectors.

    clist = []  # List of candidates.
    p = min(5, nP-1)  # Number of vectors in pruned L-curve.
    convex = False  # Are the pruned L-curves convex?

    # Sort the vectors according to the length, the longest first
    I = np.argsort(v)[::-1]

    # Main loop -- use a series of pruned L-curves The two functions
    # 'Angles' and 'Global_Behavior' are used to locate corners of the
    # pruned L-curves. Put all the corner candidates in the clist vector.
    
    while p < (nP-1) * 2:
        elmts = np.sort(I[:min(p, nP-1)])

        # First corner location algorithm
        candidate = Angles(W[elmts], elmts)
        if candidate > 0:
            convex = True
        if candidate and candidate not in clist:
            clist.append(candidate)

        # Second corner location algorithm
        candidate = Global_Behavior(P, W[elmts], elmts)
        if candidate not in clist:
            clist.append(candidate)

        p *= 2

    # Issue a warning and return if none of the pruned L-curves are convex.
    if not convex:
        k_corner = []
        info += 100
        print('Warning: Lack of convexity')
        return k_corner, info

    # Put rightmost L-curve point in clist if not already there; this is
    # used below to select the corner among the corner candidates.
    if 1 not in clist:
        clist.insert(0, 1)

    # Sort the corner candidates in increasing order.
    clist.sort()

    # Select the best corner among the corner candidates in clist.
    # The philosophy is: select the corner as the rightmost corner candidate
    # in the sorted list for which going to the next corner candidate yields
    # a larger increase in solution (semi)norm than decrease in residual norm,
    # provided that the L-curve is convex in the given point. If this is never
    # the case, then select the leftmost corner candidate in clist.
    vz = np.where(np.diff(P[clist, 1]) >= np.abs(np.diff(P[clist, 0])))[0]  # Points where the increase in solution； (semi)norm is larger than or equal to the decrease in residual norm.
    # print("type(vz):",vz)
    
    if len(vz) > 1:
        if vz[0] == 0:
            vz = vz[1:]
    elif len(vz) == 1:
        if vz[0] == 0:
            vz = np.array([])#[]
    # print("type(vz)1:",vz)
    if not vz.size:        
        # No large increase in solution (semi)norm is found and the
        # leftmost corner candidate in clist is selected.
        index = clist[-1]
    else:
        # The corner is selected as described above.
        vects = np.diff(P[clist], axis=0)
        vects = diags(1.0 / np.sqrt(np.sum(vects**2, axis=1))) @ vects
        delta = np.cross(vects[:-1], vects[1:])
        vv = np.where(delta[vz-1] <= 0)[0]
        if vv.size == 0:
            index = clist[vz[-1]]
        else:
            index = clist[vz[vv[0]]]
    
    # Corner according to original vectors without Inf, NaN, and zeros removed.
    k_corner = kept[index]

    # Plot the L-curve if required
    if fig: # Show log-log L-curve and indicate the found corner.
        plt.figure()
    
        # Calculate half the range for scaling the axes
        diffrho2 = (np.max(P[:, 0]) - np.min(P[:, 0])) / 2
        diffeta2 = (np.max(P[:, 1]) - np.min(P[:, 1])) / 2
    
        # Plot the log-log L-curve
        plt.loglog(rho, eta, 'k--o')
        plt.grid(True, which="both", ls="--")
        plt.axis('square')
        # Mark the corner
        plt.loglog([np.min(rho) / 100, rho[index]], [eta[index], eta[index]], ':r')
        plt.loglog([rho[index], rho[index]], [np.min(eta) / 100, eta[index]], ':r')
    
        # Scale axes to the same number of decades
        if abs(diffrho2) > abs(diffeta2):
            ax1 = np.min(P[:, 0])
            ax2 = np.max(P[:, 0])
            mid = np.min(P[:, 1]) + (np.max(P[:, 1]) - np.min(P[:, 1])) / 2
            ax3 = mid - diffrho2
            ax4 = mid + diffrho2
        else:
            ax3 = np.min(P[:, 1])
            ax4 = np.max(P[:, 1])
            mid = np.min(P[:, 0]) + (np.max(P[:, 0]) - np.min(P[:, 0])) / 2
            ax1 = mid - diffeta2
            ax2 = mid + diffeta2
    
        ax = [ax1 / 2, ax2, ax3, ax4]
        ax = 10 ** np.array(ax)
        plt.axis(ax)
    
        plt.xlabel(r'$|| A x - b ||_2$')
        plt.ylabel(r'$||x ||_{RKHS}$')
        plt.title(f'L-curve for {rkhs_type}, corner at {k_corner}')
        plt.show()

    return k_corner, info

# =========================================================================
# First corner finding routine -- based on angles
def Angles(W, kv):
    # Wedge products
    delta = np.cross(W[:-1], W[1:])

    mm = np.min(delta)
    if mm < 0:
        index = kv[np.argmin(delta)] + 1
    else:              #  If there is no corner, return 0.
        index = 0
    return index

# =========================================================================
# Second corner finding routine -- based on global behavior of the L-curve
def Global_Behavior(P, vects, elmts):
    hwedge = np.abs(vects[:, 1]) # Abs of wedge products between normalized vectors and horizontal,
                                 # i.e., angle of vectors with horizontal.
    An = np.sort(hwedge)   
    In = np.argsort(hwedge)  # Sort angles in increasing order.
    
    # Locate vectors for describing horizontal and vertical part of L-curve.
    count = 1
    ln = len(In)
    mn = In[0]
    mx = In[-1]
    while mn >= mx:
        mx = max(mx, In[ln-count])
        count += 1
        mn = min(mn, In[count])
    
    if count > 1:
        I = 0
        J = 0
        for i in range(count):
            for j in range(ln-1, ln-count, -1):
                if In[i] < In[j]:
                    I = In[i]
                    J = In[j]
                    break
            if I > 0:
                break
    else:
        I = In[0]
        J = In[-1]
    
    # Find intersection that describes the "origin".
    x3 = P[elmts[J]+1, 0] + (P[elmts[I], 1] - P[elmts[J]+1, 1]) / (P[elmts[J]+1, 1] - P[elmts[J], 1]) * (P[elmts[J]+1, 0] - P[elmts[J], 0])
    origin = [x3, P[elmts[I], 1]]
    
    # Find distances from the original L-curve to the "origin".  The corner
    # is the point with the smallest Euclidian distance to the "origin".
    dists = (origin[0] - P[:, 0])**2 + (origin[1] - P[:, 1])**2
    index = np.argmin(dists)
    return index