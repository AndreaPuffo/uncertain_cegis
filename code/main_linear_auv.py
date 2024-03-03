# taken from
# Systematic Synthesis of Passive Fault-Tolerant Augmented Neural
# Lyapunov Control Laws for Nonlinear Systems
# by Grande davide et al.

import cvxpy as cp
import numpy as np
import scipy as sc
import tqdm
from scipy.optimize import direct, Bounds
import matplotlib.pyplot as plt


# continuous time, Real{eig} < 0
# x1dot =  (− X_u x1 + F_1 cos α + F2 cos α  ) / m
# x2dot = (−N_r x2 + F_1 l_1 sin γ − F_2 l_2 sin γ − F_3 l_3) / J

mass, J = 500., 300.
Xu, Nr = 6.106, 210.
alpha, gamma = np.deg2rad(20.), np.deg2rad(50.)
l1, l2, l3 = 1.25, 1.25, 0.75


def build_hurwitz(A, B, P, K):
    return (A + B @ K) @ P + P @ (A + B @ K).T

def build_B(health):
    # this becomes a piecewise affine problem
    # todo: is this covered by the theory???
    health1, health2, health3 = 1., 1., 1.

    # this switch flag helps the continuity
    up = False
    if np.round(health) > health:
        up = True

    if up:
        if health < 1:
            health1 = health
        elif health < 2:
            health2 = health-1.
        elif health3 <= 3:
            health3 = health-2.
        else:
            raise ValueError
    else:
        if health <= 1:
            health1 = health
        elif health <= 2:
            health2 = health-1.
        elif health3 <= 3:
            health3 = health-2.
        else:
            raise ValueError

    return np.array([
        [health1 * np.cos(alpha) / mass, health2 * np.cos(alpha) / mass, 0.],
        [health1 * l1 * np.sin(gamma) / J, -l2 * np.sin(gamma) / J * health2, -l3 / J * health3]
    ])


def min_hurwi_eig(values, *args):
    # build matrix B
    B = build_B(values[0])
    A, P, K = args
    # we want (A+BK)P + P(A+BK)^T to have eigs < 0, but change sign for Minimization
    # robust: (A+BK)P + P(A+BK)^T + epsi * eye < 0
    iota = 0.
    hurwi = - (A + B @ K) @ P - P @ (A + B @ K).T - iota * np.eye(A.shape[0])
    return np.min(sc.linalg.eigvals(hurwi).real)

def min_cloop_eig(values, *args):
    # build matrix B
    B = build_B(values[0])
    A, P, K = args
    # we want (A+BK) to have eigs < 0, but change sign for Minimization
    cloop = - (A + B @ K)
    return np.min(sc.linalg.eigvals(cloop).real)

def closest_vertex(values, lb, ub):
    # finds the closest vertex of the convex hull given the values
    return build_B(values[0])


# uncertain matrix setting
n = 2
m = 3
epsi = 1.
eta = 10.

A = np.array([
    [- Xu/mass, 0.],
    [0., -Nr/J]
])

# B is dimension (n, m)
B_max = np.array([
    [np.cos(alpha)/mass, np.cos(alpha)/mass, 0.],
    [l1 * np.sin(gamma)/J, -l2 * np.sin(gamma)/J, -l3/J]
])

B_min1 = np.array([
    [0., np.cos(alpha) / mass, 0.],
    [0., -l2 * np.sin(gamma) / J, -l3 / J]
])

B_min2 = np.array([
    [np.cos(alpha)/mass, 0., 0.],
    [l1 * np.sin(gamma)/J, 0., -l3/J]
])

B_min3 = np.array([
    [np.cos(alpha)/mass, np.cos(alpha)/mass, 0.],
    [l1 * np.sin(gamma)/J, -l2 * np.sin(gamma)/J, 0.]
])

B_avg = 0.5 * (B_max + B_min3)  # just one starting point
all_uncertain_B = [B_avg, B_min1, B_min2, B_min3, B_max]


######################
# tentative cegis
######################

found_lyap = False
sampled_B = [B_avg]
iteration = 0

while not found_lyap:
    print('-'*80)
    print(f'Iteration {iteration}')
    print('-' * 80)
    iteration += 1

    P = cp.Variable((n,n), symmetric=True)
    W = cp.Variable((m,n))

    uncertain_constraints = []
    for idx in range(len(sampled_B)):
        M = P @ A.T + A @ P + W.T @ sampled_B[idx].T + sampled_B[idx] @ W
        uncertain_constraints += [M << -epsi * np.eye(n)]

    uncertain_constraints += [P << eta*np.eye(n), P>>0.]

    prob = cp.Problem(cp.Minimize(0.), uncertain_constraints)
    prob.solve()

    if P.value is None:
        print('Optimisation failed!')
        exit()

    P_star, W_star = P.value, W.value
    K = W_star @ np.linalg.inv(P_star)
    print(f'Candidate K: {K}')

    # find a cex
    # parameters are values of A and B
    lb_ab = [0.]
    ub_ab = [float(m)]
    bounds = Bounds(lb_ab, ub_ab)

    res = direct(min_cloop_eig, bounds=bounds, args=(A, P_star, K))
    B_found = build_B(res.x[0])
    print(f'Max eigenvalue of closed loop matrix (direct): {np.max(np.linalg.eigvals(A + B_found @ K))}')
    B_found_vtx = closest_vertex(res.x, lb_ab, ub_ab)
    print(
        f'Max eigenvalue of closed loop matrix (vertex): {np.max(np.linalg.eigvals(A + B_found_vtx @ K))}')

    res = direct(min_hurwi_eig, bounds=bounds, args=(A, P_star, K))
    print(f'Max eigenvalue of variable lyapunov matrix: {-res.fun}')
    B_found = build_B(res.x[0])
    print(f'Max eigenvalue of closed loop matrix with lyapunov (direct): {np.max(np.linalg.eigvals(A + B_found @ K))}')
    B_found_vtx = closest_vertex(res.x, lb_ab, ub_ab)
    print(f'Max eigenvalue of closed loop matrix with lyapunov (vertex): {np.max(np.linalg.eigvals(A + B_found_vtx @ K))}')

    if res.fun <= 0:
        print('found cex')
        B_cex = build_B(res.x)
        sampled_B += [B_cex]
    else:
        print('valid Lyap')
        found_lyap = True

        # sanity check. compute all possible A+BK, check if eigs are left side of plane
        max_eig = -np.inf
        print('Checking eigenvalues of closed loop matrix from all corners of convex hull...')
        for idx in tqdm.tqdm(range(len(all_uncertain_B))):
            cl = A + all_uncertain_B[idx] @ K
            # print(np.linalg.eigvals(cl))
            if np.max(np.linalg.eigvals(cl).real) > max_eig:
                max_eig = np.max(np.linalg.eigvals(cl))

            if not all(np.linalg.eigvals(cl).real < 0.):
                print(f'Found eig > 0 with couple: {A}, {all_uncertain_B[idx]}')

        print(f'Max closed loop eigenvalue of vertices polytope: {max_eig}')


