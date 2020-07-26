import numpy as np
import cvxpy as cp
from scipy.special import erfc, erf

def RBFKernel(dt, shift, sigma=1.):
    return np.sqrt(np.pi/2)*sigma*(erfc((shift-dt)/(np.sqrt(2)*sigma)) - erfc(shift/(np.sqrt(2)*sigma)))

def HGen(ti, tj, alphaji, nbDistSample=1):
    H=0
    sigma = 1
    for i, shift in enumerate([i for i in range(1, nbDistSample+1)]):
        H += alphaji[i] * np.exp(-((ti - tj - shift) ** 2) / (2 * (sigma ** 2)))
    return H

def H(ti, tj, alphaji, nbDistSample=1):
    H=0
    sigma=1
    H += alphaji[0]
    for i, shift in enumerate([i for i in range(1, nbDistSample)]):
        H += alphaji[i] * np.exp(-((ti-tj-shift)**2) / (2*(sigma**2)))
    return H

    if nbDistSample == 3 and True:
        H += alphaji[0]
        H += alphaji[1]*(ti-tj)
        H += alphaji[2]/(ti-tj)
        return H

    if nbDistSample == 1 and True:
        H += alphaji[0] / (ti-tj)
        return H

    if nbDistSample == 2 and True:
        H += alphaji[0]
        H += alphaji[1]/(ti-tj)
        return H

def logF(ti, tj, alphaji, nbDistSample=1):
    S = 0
    S += cp.log(H(ti, tj, alphaji, nbDistSample)+1e-20) + logS(ti, tj, alphaji, nbDistSample)

    return S
