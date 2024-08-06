import numpy as np
import cvxpy as cp


"""
Constraints for Lemma 2 or Lemma 3
"""
def constraints_lemma23(As, Bs, Bws, Es, L, umax, tau, K=None):

    n = As[0].shape[0]
    s = len(As)
    assert len(As) == len(Bs) # sanity check
    m = Bs[0].shape[1]
    combinations = int(2**m)
    h = L.shape[0]


    Q = cp.Variable((n, n), symmetric=True)  # Q is the stability ellipsoid
    if K is None:
        Y = cp.Variable((m, n))  # Y = KQ   K is (m,n), Q is (n,n) --> (m,n)
    else:
        Y = K @ Q  # Y = KQ   K is (m,n), Q is (n,n) --> (m,n)
    Z = cp.Variable((m, n))  # Z = HQ

    control_constraints = []
    for i in range(s):
        for j in range(combinations):
            tmp = As[i] @ Q + Bs[i] @ Es[j] @ Y + Bs[i] @ (np.eye(m) - Es[j]) @ Z
            control_constraints += [
                cp.bmat([
                    [tau * Q,           np.zeros((n, n)).T,         tmp.T],
                    [np.zeros((n, n)),  (1. - tau) * np.eye(n),     Bws[i].T],
                    [tmp,               Bws[i],                     Q]
                ]) >> 0.
            ]

    ## state constraints
    state_constraints = []
    # NOTA Code Syntax:  L[None, i, :] makes the dimension of row_i of L (1 x n)
    # using only L[i,:] gives a matrix of dimension (n,) which has issues with dimensionality of the block diag
    # similarly for np.eye(1) instead of simply using 1.
    for i in range(h):
        state_constraints += [
            cp.bmat([
                [np.eye(1),             L[None, i, :] @ Q],
                [Q @ L[None, i, :].T,   Q]
            ]) >> 0.
        ]

    ## input constraints
    input_constraints = []
    for j in range(m):
        input_constraints += [
            cp.bmat([
                [umax ** 2 * np.eye(1),     Z[None, j, :]],
                [Z[None, j, :].T,           Q]
            ]) >> 0.
        ]

    return Q, Y, Z, control_constraints + state_constraints + input_constraints


"""
this follows the approach on Interpolation based predictive control by ellipsoidal invariant sets
Daniel Rubin, Pedro Mercader, Per-Olof Gutman, Hoai-Nam Nguyen, Alberto Bemporad

the model follows 
x_{k+1} = A_k x_k + B_k u_k + D_k w_k
x \in X = { |Lx| < 1}
u \in U = { |u| < u_max }
w \in W = { ||w||_2 < 1 }
A(k) = \sum_{i=1}^s  alpha_i(k) * A_i
B(k) = \sum_{i=1}^s  beta_i(k) * B_i
D(k) = \sum_{i=1}^s  deta_i(k) * D_i

"""