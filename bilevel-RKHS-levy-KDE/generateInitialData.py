import numpy as np


def generate_initialsamples(length, dt, jump_prob=0.05, big_scale=1.0):
    # Decide variances: pick big or small
    variances = np.where(
        np.random.uniform(0,1,size=length) < jump_prob,
        big_scale**2,
        dt
    )
    # Directly sample from normal with chosen variance
    increments = np.random.normal(0, np.sqrt(variances))
    # Build the path
    path = np.cumsum(increments)
    return path


def generate_piecewise_constant(length, segments, values):
    X0 = np.zeros(length)
    start = 0
    for end, val in zip(segments + [length], values):
        X0[start:end] = val
        start = end
    return X0

