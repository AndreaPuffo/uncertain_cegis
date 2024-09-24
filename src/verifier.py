import numpy as onp
import scipy


def verifierBemporad(P, K, H, computeAB, paraSize, matrixbounds, mask, tau):

    valid = False
    newVertices = []

    stateSize, inputSize = P.shape[0], K.shape[0]
    Q = onp.linalg.inv(P)

    # todo: can we not simply check A+BK?
    costEig = lambda x: costEigPKBemporad(x, Q, K, H, computeAB, paraSize, mask, tau)
    b = scipy.optimize.Bounds(mask.T @ matrixbounds.lb, mask.T @ matrixbounds.ub)
    # res=scipy.optimize.direct(costEig,bounds=b,locally_biased=True)
    res = scipy.optimize.shgo(costEig, bounds=b, options={"f_tol": 1e-6})
    # res=scipy.optimize.minimize(costEig,res.x,bounds=Bounds)
    # halo = HALO(costEig, [[Bounds.lb[i],Bounds.ub[i]] for i in range(0,len(Bounds.lb))], max_feval, max_iter, beta, local_optimizer, verbose)
    # result=halo.minimize();
    result = {'best_f': res.fun, 'best_x': res.x}
    # break
    print("Verifier says: ", result['best_f'])
    # todo: is this correct? should it be >= +1e-9?
    if result['best_f'] >= -1e-9:
        valid = True
    else:
        x = result['best_x']
        x = mask @ x
        x = onp.reshape(x, (1, stateSize + inputSize + 1))
        xState = x[0:1, 0:stateSize]
        u = x[0:1, stateSize:stateSize + inputSize]
        p = x[0:1, stateSize + inputSize:]
        for k in range(paraSize):
            para_v = onp.ones((paraSize, ))
            para_v[k] = p[0:1]
            newVertices += [computeAB(xState, u, para_v)]

    return newVertices, valid


def costEigPKBemporad(x, Q, K, H, computeAB, paraSize, mask, tau):

    stateSize, inputSize = Q.shape[0], K.shape[0]

    x = onp.reshape(x, (mask.shape[1], 1))
    # A,B=approximateOnPointJAC(x,nbrs)
    x = mask @ x
    x = x.reshape((1, stateSize + inputSize + 1))
    xState = x[0:1, 0:stateSize]
    u = x[0:1, stateSize:stateSize + inputSize]
    p = x[0:1, stateSize + inputSize:]
    # xTT=system.forward(u,xState,p)
    # overallDataset+=[(xState.T,u[0],xTT,computeAB(xState.T,u))]
    v = onp.Inf

    Y = K @ Q
    Z = H @ Q
    Es = []

    Dw = onp.eye(stateSize) * 0
    combinations = int(2 ** inputSize)
    for j in range(combinations):
        Es += [onp.diag([int(b) for b in onp.binary_repr(j, inputSize)])]

    for k in range(paraSize):
        para_v = onp.ones((paraSize, 1))
        para_v[k] = p[0:1]
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
            v = onp.minimum(onp.min(onp.real(eig)), v)

    return v

def verifierKhotare(P, K, computeAB, paraSize, matrixbounds, mask, tau):

    valid = False
    newVertices = []

    stateSize, inputSize = P.shape[0], K.shape[0]

    costEig = lambda x: costEigPKhotare(x, P, K, computeAB, paraSize, mask, tau)

    res = scipy.optimize.direct(costEig, bounds=matrixbounds, locally_biased=True)
    # res=scipy.optimize.minimize(costEig,res.x,bounds=Bounds)
    # halo = HALO(costEig, [[Bounds.lb[i],Bounds.ub[i]] for i in range(0,len(Bounds.lb))], max_feval, max_iter, beta, local_optimizer, verbose)
    # result=halo.minimize();
    result = {'best_f': res.fun, 'best_x': res.x}
    print(result['best_f'])
    if result['best_f'] >= 0:
        valid = True
    else:
        x = result['best_x']
        x = onp.reshape(x, (1, stateSize + inputSize + 1))
        xState = x[0:1, 0:stateSize]
        u = x[0:1, stateSize:stateSize + inputSize]
        p = x[0:1, stateSize + inputSize:]
        for k in range(paraSize):
            para_v = onp.ones((paraSize))
            para_v[k] = p[0:1]
            newVertices += [computeAB(xState, u, para_v)]

    return newVertices, valid

def costEigPKhotare(x, P, K, computeAB, paraSize, mask, tau):

    stateSize, inputSize = P.shape[0], K.shape[0]

    x = onp.reshape(x, (1, stateSize + inputSize + 1))
    # A,B=approximateOnPointJAC(x,nbrs)
    xState = x[0:1, 0:stateSize]
    u = x[0:1, stateSize:stateSize + inputSize]
    p = x[0:1, stateSize + inputSize:]
    # xTT=system.forward(u,xState,p)
    # overallDataset+=[(xState.T,u[0],xTT,computeAB(xState.T,u))]
    v = onp.Inf
    for k in range(paraSize):
        para_v = onp.ones((paraSize, 1))
        para_v[k] = p[0:1]
        A, B = computeAB(xState.T, u, para_v)
        eig = scipy.sparse.linalg.eigsh(
            onp.vstack([onp.hstack([P, (A + B @ K).T @ P]),
                        onp.hstack([P @ (A + B @ K), P])]), 1, which='SA', return_eigenvectors=False)
        # print(eig)
        v = onp.minimum(onp.min(onp.real(eig)), v)
    return v


def verifierEllipsoid(P, K, H, computeAB, paraSize, matrixbounds, mask, tau):

    valid = False
    newVertices = []

    stateSize, inputSize = P.shape[0], K.shape[0]

    Q = onp.linalg.inv(P)
    costEig = lambda x: costEigPKEllipsoid(x, Q, K, H)
    b = scipy.optimize.Bounds(mask.T @ matrixbounds.lb, mask.T @ matrixbounds.ub)
    res = scipy.optimize.direct(costEig, bounds=b, locally_biased=True)
    # res=scipy.optimize.minimize(costEig,res.x,vbounds=Bounds)
    # halo = HALO(costEig, [[Bounds.lb[i],Bounds.ub[i]] for i in range(0,len(Bounds.lb))], max_feval, max_iter, beta, local_optimizer, verbose)
    # result=halo.minimize();
    result = {'best_f': res.fun, 'best_x': res.x}
    # break
    print(result['best_f'])
    if result['best_f'] >= 0:
        valid = True
    else:
        x = result['best_x']
        x = mask @ x
        print("last counter example", x)
        x = onp.reshape(x, (1, stateSize + inputSize + 1))
        xState = x[0:1, 0:stateSize]
        u = x[0:1, stateSize:stateSize + inputSize]
        p = x[0:1, stateSize + inputSize:]
        for k in range(paraSize):
            para_v = onp.ones((paraSize))
            para_v[k] = p[0:1]
            newVertices += [computeAB(xState, u, para_v)]

    return newVertices, valid


def costEigPKEllipsoid(x, Q, K, H, computeAB, paraSize, mask, tau):

    stateSize, inputSize = Q.shape[0], K.shape[0]

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
    v = onp.Inf

    Y = K @ Q
    Z = H @ Q
    Es = []

    Dw = onp.eye(stateSize) * 0
    combinations = int(2 ** inputSize)
    for j in range(combinations):
        Es += [onp.diag([int(b) for b in onp.binary_repr(j, inputSize)])]

    for k in range(paraSize):
        para_v = onp.ones((paraSize, 1))
        para_v[k] = p[0:1]
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
            v = onp.minimum(onp.min(onp.real(eig)), v)

    return v