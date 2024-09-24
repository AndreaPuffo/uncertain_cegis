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

def computeEllipsoid(Kext):
    import cvxpy as cp

    x = onp.zeros((stateSize)) + 1;
    u = onp.zeros((inputSize)) + 1
    p = onp.ones((paraSize))
    setOfVertices = [computeAB(x, u, p)]
    print(setOfVertices)

    while (True):
        def synthesizeController():
            nonlocal Kext
            global stateSize
            global inputSize
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
                    cp.vstack([cp.hstack([onp.eye(1) * (Bounds.ub[stateSize + k] ** 2), Zr]),
                               cp.hstack([Zr.T, Q])]) >> 0
                ]

            L = onp.diag(1 / Bounds.ub[:stateSize])
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
            P = onp.linalg.inv(Q.value);
            K = (Y.value @ P)
            H = Z.value @ P
            for AB in setOfVertices:
                A, B = AB
                print(onp.linalg.eigvals(A + B @ K))
                assert (onp.all(onp.abs(onp.linalg.eigvals(A + B @ K)) < 1))
            return P, K, H

        P, K, H = synthesizeController()

        def costEigPK(x, Q, K, H):
            global mask
            x = onp.reshape(x, (mask.shape[1], 1))
            # A,B=approximateOnPointJAC(x,nbrs)
            x = mask @ x
            x = x.reshape((1, stateSize + inputSize + 1))
            # A,B=approximateOnPointJAC(x,nbrs)
            xState = x[0:1, 0:stateSize]
            u = x[0:1, stateSize:stateSize + inputSize]
            p = x[0:1, stateSize + inputSize:]
            # xTT=system.forward(u,xState,p)
            # overallDataset+=[(xState.T,u[0],xTT,computeAB(xState.T,u))]
            v = onp.Inf;

            Y = K @ Q
            Z = H @ Q
            Es = []

            Dw = onp.eye(stateSize) * 0
            combinations = int(2 ** inputSize)
            for j in range(combinations):
                Es += [onp.diag([int(b) for b in onp.binary_repr(j, inputSize)])]

            for k in range(paraSize):
                para_v = onp.ones((paraSize, 1));
                para_v[k] = p[0:1];
                A, B = computeAB(xState.T, u, para_v)
                for Ej in Es:
                    ACL = (A @ Q + B @ Ej @ Y + B @ (onp.eye(inputSize) - Ej) @ Z)
                    cond = onp.vstack([onp.hstack([(tau) * Q, Q * 0, ACL.T]),
                                       onp.hstack([Q * 0, onp.eye(stateSize) * (1 - tau), Dw]),
                                       onp.hstack([ACL, Dw.T, Q]), ])
                    # print(constraints[-1].shape)
                    # cond=scipy.linalg.block_diag(constraints)
                    eig = scipy.sparse.linalg.eigsh(cond, 1, which='SA', return_eigenvectors=False)
                    # print(eig)
                    v = min(min(onp.real(eig)), v)

            return v

        Q = onp.linalg.inv(P)
        costEig = lambda x: costEigPK(x, Q, K, H)
        b = scipy.optimize.Bounds(mask.T @ Bounds.lb, mask.T @ Bounds.ub)
        res = scipy.optimize.direct(costEig, bounds=b, locally_biased=True)
        # res=scipy.optimize.minimize(costEig,res.x,vbounds=Bounds)
        # halo = HALO(costEig, [[Bounds.lb[i],Bounds.ub[i]] for i in range(0,len(Bounds.lb))], max_feval, max_iter, beta, local_optimizer, verbose)
        # result=halo.minimize();
        result = {'best_f': res.fun, 'best_x': res.x}
        # break
        print(result['best_f'])
        if result['best_f'] >= 0:
            break
        else:
            x = result['best_x']
            x = mask @ x
            print("last counter example", x)
            x = onp.reshape(x, (1, stateSize + inputSize + 1))
            xState = x[0:1, 0:stateSize]
            u = x[0:1, stateSize:stateSize + inputSize]
            p = x[0:1, stateSize + inputSize:]
            for k in range(paraSize):
                para_v = onp.ones((paraSize));
                para_v[k] = p[0:1];
                setOfVertices += [computeAB(xState, u, para_v)]
    return P, K

