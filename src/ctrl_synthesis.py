import numpy as onp
import cvxpy as cp
import scipy


def synthesizeBemporadController(stateSize, inputSize, matrixbounds, setOfVertices, tau):

    # define the optimisation variables
    Q = cp.Variable((stateSize, stateSize), symmetric=True)
    Dw = onp.eye(stateSize) * 0
    Y = cp.Variable((inputSize, stateSize))
    Z = cp.Variable((inputSize, stateSize))

    constraints = []
    for k in range(inputSize):
        Zr = Z[k:k + 1, :]
        # print(Zr.shape)
        constraints += [
            cp.vstack([cp.hstack([onp.eye(1) * (matrixbounds.ub[stateSize + k] ** 2), Zr]),
                       cp.hstack([Zr.T, Q])]) >> 0
        ]

    L = onp.diag(1 / matrixbounds.ub[:stateSize])
    for k in range(stateSize):
        Lr = L[k:k + 1, :]
        # print(Lr.shape)
        constraints += [
            cp.vstack([cp.hstack([onp.eye(1), Lr @ Q]),
                       cp.hstack([Q @ Lr.T, Q])]) >> 0
        ]

    combinations = int(2 ** inputSize)
    Es = []

    for j in range(combinations):
        Es += [onp.diag([int(b) for b in onp.binary_repr(j, inputSize)])]
    for AB in setOfVertices:
        A, B = AB
        for Ej in Es:
            ACL = (A @ Q + B @ Ej @ Y + B @ (onp.eye(inputSize) - Ej) @ Z)
            constraints += [
                cp.vstack([cp.hstack([(tau) * Q, Q * 0, ACL.T]),
                           cp.hstack([Q * 0, onp.eye(stateSize) * (1 - tau), Dw]),
                           cp.hstack([ACL, Dw.T, Q]),
                           ]) >> 1e-4

            ]
    prob = cp.Problem(cp.Minimize(-cp.trace(Q)), constraints)
    prob.solve(solver='MOSEK', verbose=False)

    print("The program optimal value [trace(Q)] is", prob.value)
    # print("A solution X is")
    P = onp.linalg.inv(Q.value)
    K = (Y.value @ P)
    H = Z.value @ P
    max_eigval = 0.
    for AB in setOfVertices:
        A, B = AB
        tmp = onp.max(onp.absolute(onp.linalg.eigvals(A + B @ K)))
        if tmp > max_eigval:
            max_eigval = tmp
        assert (onp.all(onp.abs(onp.linalg.eigvals(A + B @ K)) < 1))
    print(f'Max Norm A+BK found in learner: {max_eigval}')
    return P, K, H


# %%
def synthesizeKhotareController(stateSize, inputSize, matrixbounds, setOfVertices, tau):
    Qx = onp.eye(stateSize) / 1
    R = onp.eye(inputSize) / 1  # sqrt in realtà

    W = cp.Variable((stateSize, stateSize), symmetric=True)
    X = cp.Variable((inputSize, inputSize), symmetric=True)
    Q = cp.Variable((inputSize, stateSize))
    gamma = cp.Variable((1, 1))

    A, B = setOfVertices[0]
    constraints = [W >> 1 * onp.eye(stateSize),
                   W << 100 * onp.eye(stateSize),
                   # cp.vstack([ cp.hstack([onp.eye(1),      onp.ones((stateSize,1)).T]),
                   #             cp.hstack([onp.ones((stateSize,1)),W])
                   #   ])>>0
                   # cp.vstack([ cp.hstack([-X,      (R@Q)]),
                   #             cp.hstack([(R@Q).T, -W])
                   #   ])<<0
                   ]
    L = onp.diag(1 / matrixbounds.ub[:stateSize])
    for k in range(stateSize):
        Lr = L[k:k + 1, :]
        print(Lr.shape)
        constraints += [
            cp.vstack([cp.hstack([onp.eye(1), Lr @ W]),
                       cp.hstack([W @ Lr.T, W])]) >> 0
        ]

    for AB in setOfVertices:
        A, B = AB
        constraints += [
            # cp.vstack([cp.hstack([-W+onp.eye(stateSize),      A@W+B@Q]),
            #             cp.hstack([(A@W+B@Q).T,-W])
            #   ])<<-1e-2
            cp.vstack([cp.hstack([W, (A @ W + B @ Q).T, W @ Qx, Q.T @ R]),
                       cp.hstack([(A @ W + B @ Q), W, W * 0, Q.T * 0]),
                       cp.hstack([(W @ Qx).T, W * 0, gamma * onp.eyMastie(stateSize), Q.T * 0]),
                       cp.hstack([(Q.T @ R).T, Q * 0, Q * 0, gamma * onp.eye(inputSize)])
                       ]) >> 1e-2

        ]
    prob = cp.Problem(cp.Minimize(gamma / 1000), constraints)
    prob.solve(solver='MOSEK', verbose=True)

    print("The optimal value is", prob.value)
    # print("A solution X is")
    P = onp.linalg.inv(W.value)
    K = (Q.value @ onp.linalg.inv(W.value))

    for AB in setOfVertices:
        A, B = AB
        print(onp.linalg.eigvals(A + B @ K))
        assert (onp.all(onp.abs(onp.linalg.eigvals(A + B @ K)) < 1))
    return P, K


def synthesizeEllipsoidController(Kext, stateSize, inputSize, matrixbounds, setOfVertices, tau):

    Q = cp.Variable((stateSize, stateSize), symmetric=True)
    Dw = onp.eye(stateSize) * 0
    Y = cp.Variable((inputSize, stateSize))
    Z = cp.Variable((inputSize, stateSize))

    # print((Q@Kext).shape)
    constraints = [Kext @ Q == Y]
    for k in range(inputSize):
        Zr = Z[k:k + 1, :]
        print(Zr.shape)
        constraints += [
            cp.vstack([cp.hstack([onp.eye(1) * (matrixbounds.ub[stateSize + k] ** 2), Zr]),
                       cp.hstack([Zr.T, Q])]) >> 0
        ]

    L = onp.diag(1 / matrixbounds.ub[:stateSize])
    for k in range(stateSize):
        Lr = L[k:k + 1, :]
        print(Lr.shape)
        constraints += [
            cp.vstack([cp.hstack([onp.eye(1), Lr @ Q]),
                       cp.hstack([Q @ Lr.T, Q])]) >> 1e-4
        ]

    combinations = int(2 ** inputSize)
    Es = []

    for j in range(combinations):
        Es += [onp.diag([int(b) for b in onp.binary_repr(j, inputSize)])]
    for AB in setOfVertices:
        A, B = AB
        for Ej in Es:
            ACL = (A @ Q + B @ Ej @ Y + B @ (onp.eye(inputSize) - Ej) @ Z)
            constraints += [
                cp.vstack([cp.hstack([(tau) * Q, Q * 0, ACL.T]),
                           cp.hstack([Q * 0, onp.eye(stateSize) * (1 - tau), Dw]),
                           cp.hstack([ACL, Dw.T, Q]),
                           ]) >> 1e-4

            ]
    prob = cp.Problem(cp.Minimize(-cp.trace(Q)), constraints)
    prob.solve(solver='MOSEK', verbose=True)

    print("The optimal value is", prob.value)
    # print("A solution X is")
    P = onp.linalg.inv(Q.value)
    K = (Y.value @ P)
    H = Z.value @ P
    for AB in setOfVertices:
        A, B = AB
        print(onp.linalg.eigvals(A + B @ K))
        assert (onp.all(onp.abs(onp.linalg.eigvals(A + B @ K)) < 1))
    return P, K, H

def controllerH2(stateSize, inputSize, paraSize, computeAB):
    # pass

    x = onp.zeros((stateSize, ))
    u = onp.zeros((inputSize, ))
    p = onp.ones((paraSize, ))
    setOfVertices = [computeAB(x, u, p)]

    def synthesizeController():
        Qx = onp.eye(stateSize) * 1e1
        R = onp.eye(inputSize) / 1e1  # sqrt in realtà

        X = cp.Variable((stateSize, stateSize), symmetric=True)
        Y = cp.Variable((inputSize, inputSize), symmetric=True)
        W = cp.Variable((inputSize, stateSize))
        gamma = cp.Variable((1, 1))

        A, B = setOfVertices[0]
        constraints = [X >> onp.eye(stateSize),  # Y>>0, gamma<=0  ,
                       cp.vstack([cp.hstack([-Y, (R @ W)]),
                                  cp.hstack([(R @ W).T, -X])
                                  ]) << 0,
                       cp.vstack([cp.hstack([-X + onp.eye(stateSize), A @ X + B @ W]),
                                  cp.hstack([(A @ X + B @ W).T, -X])
                                  ]) << 0

                       ]
        prob = cp.Problem(cp.Minimize(cp.trace(Qx @ X @ Qx.T) + cp.trace(Y)), constraints)
        prob.solve(solver='MOSEK', verbose=True)

        print("The optimal value is", prob.value)
        # print("A solution X is")
        P = onp.linalg.inv(X.value)
        K = (W.value @ P)

        for AB in setOfVertices:
            A, B = AB
            print(onp.linalg.eigvals(A + B @ K))
            assert (onp.all(onp.abs(onp.linalg.eigvals(A + B @ K)) < 1))
        return P, K

    P, K = synthesizeController()

    return P, K
