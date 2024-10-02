import numpy as onp
import scipy
from ctrl_synthesis import synthesizeBemporadController, synthesizeKhotareController, synthesizeEllipsoidController
from verifier import verifierBemporad, verifierKhotare, verifierEllipsoid


def CegisBemporad(benchmark, computeAB, variableBounds, mask, tau=1. - 1e-4):
    import cvxpy as cp

    stateSize, inputSize, paraSize = benchmark.stateSize, benchmark.inputSize, benchmark.paraSize

    x = onp.zeros((stateSize,)) + 1
    u = onp.zeros((inputSize,)) + 1
    p = onp.ones((paraSize,))
    setOfVertices = [computeAB(x, u, p)]
    # print(setOfVertices)

    found = False
    max_cegis_iters = 100
    itr = 1
    while not found and itr < max_cegis_iters:

        print('='*80)
        print(f'CEGIS Iteration {itr}')

        # learner
        P, K, H = synthesizeBemporadController(stateSize, inputSize, variableBounds, setOfVertices, tau)

        # verifier
        newVertices, found = verifierBemporad(P, K, H, computeAB, paraSize, variableBounds, mask, tau)
        setOfVertices = setOfVertices + newVertices

        # update iteration counter
        itr += 1

    return P, K


def CegisKhotare(benchmark, computeAB, variableBounds, mask, tau=1. - 1e-4):
    from halo import HALO

    max_feval = 8000  # maximum number of function evaluations
    max_iter = 8000  # maximum number of iterations
    beta = 1e-2  # beta controls the usage of the local optimizers during the optimization process
    # With a lower value of beta HALO will use the local search more rarely and viceversa.
    # The parameter beta must be less than or equal to 1e-2 or greater than equal to 1e-4.
    local_optimizer = 'L-BFGS-B'  # Choice of local optimizer from scipy python library.
    # The following optimizers are available: 'L-BFGS-B', 'Nelder-Mead', 'TNC' and 'Powell'.
    # For more infomation about the local optimizers please refer the scipy documentation.
    verbose = 0  # this controls the verbosity level, fixed to 0 no output of the optimization progress
    # will be printed.

    stateSize, inputSize, paraSize = benchmark.stateSize, benchmark.inputSize, benchmark.paraSize

    x = onp.zeros((stateSize,)) + 0.5
    u = onp.zeros((inputSize,)) + 0.5
    p = onp.ones((paraSize, ))
    setOfVertices = [computeAB(x, u, p)]

    found = False
    max_cegis_iters = 100
    itr = 1
    while not found and itr < max_cegis_iters:

        # learner
        P, K = synthesizeKhotareController(stateSize, inputSize, variableBounds, setOfVertices, tau)

        # verifier
        newVertices, found = verifierKhotare(P, K, computeAB, paraSize, variableBounds, mask, tau)
        setOfVertices = setOfVertices + newVertices

        # update iteration counter
        itr += 1

    return P, K


def CegisEllipsoid(Kext, benchmark, computeAB, variableBounds, mask, tau=1. - 1e-4):

    stateSize, inputSize, paraSize = benchmark.stateSize, benchmark.inputSize, benchmark.paraSize

    x = onp.zeros((stateSize)) + 1
    u = onp.zeros((inputSize)) + 1
    p = onp.ones((paraSize))
    setOfVertices = [computeAB(x, u, p)]
    print(setOfVertices)

    found = False
    max_cegis_iters = 100
    itr = 1
    while not found and itr < max_cegis_iters:

        # learner
        P, K, H = synthesizeEllipsoidController(Kext, stateSize, inputSize, variableBounds, setOfVertices, tau)
        # verifier
        newVertices, found = verifierEllipsoid(P, K, H, computeAB, paraSize, variableBounds, mask, tau)
        setOfVertices = setOfVertices + newVertices

        # update iteration counter
        itr += 1

    return P, K
