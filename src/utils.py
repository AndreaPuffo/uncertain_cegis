import numpy as np
import cvxpy as cp
import scipy as sc


def solve_saturation_ctrl_synthesis(domain_ellipse, A, B):
    """
    taken from section 2.5 of
    An analysis and design method for linear systems subject to
    actuator saturation and disturbance
    Tingshu Hua , Zongli Lina , Ben M. Chen

    :param domain_ellipse:
    :param A:
    :param B:
    :return:
    """

    n = A.shape[0]
    m = B.shape[1]
    # inv_ellipse_P = np.linalg.inv(Ellipse_P)
    # Q = rho * inv_ellipse_P
    Q = cp.Variable((n,n))
    # H = cp.Variable((m,n))
    # G = H @ Q
    G = cp.Variable((m,n))
    # Y is the new K
    Y = cp.Variable((m,n))


    # condition a) >> 0
    gamma = cp.Variable((1,1))
    condition_a = get_condition_a_constraint(gamma, domain_ellipse, Q)

    # condition b << 0
    condition_b = get_condition_b_constraint(A, B, Q, G, Y)

    # condition c
    condition_c = get_condition_c_constraint(G, Q)

    prob = cp.Problem(cp.Minimize(gamma), condition_a + condition_b + condition_c + [Q >> 0.])
    prob.solve()

    return gamma.value, Y.value, Q.value, G.value

def get_condition_a_constraint(gamma, domain_ellipse, Q):
    n = Q.shape[0]
    return [cp.bmat([
        [gamma * domain_ellipse, np.eye(n)],
        [np.eye(n), Q]
    ]) >> 0.]

def get_condition_b_constraint(A, B, Q, G, Y, eta=0.):
    n = A.shape[0]
    m = B.shape[1]
    condition_b = []
    combinations = int(2. ** m)
    for c in range(combinations):
        binary_repr = np.binary_repr(c, m)
        rows = []
        for r in range(m):
            if binary_repr[r] == '0':
                rows.append(G[r, :])
            elif binary_repr[r] == '1':
                rows.append(Y[r, :])
            else:
                raise ValueError
        feedback_matrix = cp.vstack(rows)
        condition_b.append(
            # rho * (A + B @ feedback_matrix).T @ inv_ellipse_P +
            # rho * inv_ellipse_P @ (A + B @ feedback_matrix) << 0.
            Q @ A.T + A @ Q + feedback_matrix.T @ B.T + B @ feedback_matrix << eta * np.eye(n)
        )
    return condition_b

def get_condition_c_constraint(G, Q):
    condition_c = []
    m = G.shape[0]
    for i in range(m):
        condition_c.append(
            cp.bmat([
                [np.eye(1), G[i, :][:, None].T],
                [G[i, :][:, None], Q]
            ]) >> 0.
        )
    return condition_c


def min_hurwitz_eig(A, B, K, P):
    # we want (A+BK)P + P(A+BK)^T to have eigs < 0, but change sign for Minimization
    hurwi = - (P @ A.T + A @ P + K.T @ B.T + B @ K)
    return np.min(sc.linalg.eigvals(hurwi).real)


def min_closedloop_eig(A, B, K):
    # we want (A+BK) to have eigs < 0, but change sign for Minimization
    cloop = - (A + B @ K)
    return np.min(sc.linalg.eigvals(cloop).real)  # this actually returns the max eig of closed loop