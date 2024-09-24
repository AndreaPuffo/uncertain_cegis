import numpy as onp
import scipy
from ctrl_synthesis import synthesizeBemporadController
from verifier import verifier


def CegisBemporad(benchmark, computeAB, matrixbounds, mask, tau=1. - 1e-4):
    import cvxpy as cp

    stateSize, inputSize, paraSize = benchmark.stateSize, benchmark.inputSize, benchmark.paraSize

    x = onp.zeros((stateSize,)) + 1
    u = onp.zeros((inputSize,)) + 1
    p = onp.ones((paraSize,))
    setOfVertices = [computeAB(x, u, p)]
    print(setOfVertices)

    found = False
    max_cegis_iters = 100
    itr = 1
    while not found and itr < max_cegis_iters:
        # learner
        P, K, H = synthesizeBemporadController(stateSize, inputSize, matrixbounds, setOfVertices, tau)

        # verifier
        newVertices, found = verifier(P, K, H, computeAB, paraSize, matrixbounds, mask, tau)
        setOfVertices = setOfVertices + newVertices

        # update iteration counter
        itr += 1

    return P, K
