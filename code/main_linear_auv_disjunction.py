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

def build_B1(health):
    return np.array([
        [health * np.cos(alpha) / mass,  np.cos(alpha) / mass, 0.],
        [health * l1 * np.sin(gamma) / J, -l2 * np.sin(gamma) / J, -l3 / J]
    ])

def build_B2(health):
    return np.array([
        [ np.cos(alpha) / mass,  health*np.cos(alpha) / mass, 0.],
        [ l1 * np.sin(gamma) / J, -l2 * health *np.sin(gamma) / J, -l3 / J]
    ])
def build_B3(health):
    return np.array([
        [np.cos(alpha) / mass,  np.cos(alpha) / mass, 0.],
        [l1 * np.sin(gamma) / J, -l2 * np.sin(gamma) / J, -l3 * health / J]
    ])

def min_hurwi_eig(values, *args):
    # build matrix B
    B1 = build_B1(values[0])
    B2 = build_B2(values[1])
    B3 = build_B3(values[2])
    A, P, K = args
    # we want (A+BK)P + P(A+BK)^T to have eigs < 0, but change sign for Minimization
    # robust: (A+BK)P + P(A+BK)^T + epsi * eye < 0
    iota = 0.
    hurwi1 = - (A + B1 @ K) @ P - P @ (A + B1 @ K).T - iota * np.eye(A.shape[0])
    hurwi2 = - (A + B2 @ K) @ P - P @ (A + B2 @ K).T - iota * np.eye(A.shape[0])
    hurwi3 = - (A + B3 @ K) @ P - P @ (A + B3 @ K).T - iota * np.eye(A.shape[0])
    return np.min([
        np.min(sc.linalg.eigvals(hurwi1).real),
        np.min(sc.linalg.eigvals(hurwi2).real),
        np.min(sc.linalg.eigvals(hurwi3).real)])


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

B_avg1 = 0.5 * (B_max + B_min1)
B_avg2 = 0.5 * (B_max + B_min2)
B_avg3 = 0.5 * (B_max + B_min3)
vertex_B = [B_avg1, B_avg2, B_avg3, B_min1, B_min2, B_min3, B_max]


######################
# tentative cegis
######################

found_lyap = False
sampled_B = [B_avg1, B_avg2, B_avg3]
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
    lb_ab = [0., 0., 0.]
    ub_ab = [1., 1., 1.]
    bounds = Bounds(lb_ab, ub_ab)

    res = direct(min_hurwi_eig, bounds=bounds, args=(A, P_star, K))
    print(f'Max eigenvalue of variable lyapunov matrix: {-res.fun}')
    B1_found, B2_found, B3_found = build_B1(res.x[0]), build_B2(res.x[1]), build_B3(res.x[2])
    print(f'Max eigenvalue of closed loop matrix with lyapunov (direct): '
          f'{np.max(np.linalg.eigvals(A + B1_found @ K))}, {np.max(np.linalg.eigvals(A + B2_found @ K))}, '
          f'{np.max(np.linalg.eigvals(A + B3_found @ K))}')
    # B_found_vtx = closest_vertex(res.x, lb_ab, ub_ab)
    # print(f'Max eigenvalue of closed loop matrix with lyapunov (vertex): {np.max(np.linalg.eigvals(A + B_found_vtx @ K))}')

    if res.fun <= 0:
        print('found cex')
        B1_found, B2_found, B3_found = build_B1(res.x[0]), build_B2(res.x[1]), build_B3(res.x[2])
        sampled_B += [B1_found, B2_found, B3_found]
    else:
        print('valid Lyap')
        found_lyap = True

        # sanity check. compute all possible A+BK, check if eigs are left side of plane
        max_eig = -np.inf
        print('Checking eigenvalues of closed loop matrix from all corners of convex hull...')
        for idx in tqdm.tqdm(range(len(vertex_B))):
            cl = A + vertex_B[idx] @ K
            # print(np.linalg.eigvals(cl))
            if np.max(np.linalg.eigvals(cl).real) > max_eig:
                max_eig = np.max(np.linalg.eigvals(cl))

            if not all(np.linalg.eigvals(cl).real < 0.):
                print(f'Found eig > 0 with couple: {A}, {vertex_B[idx]}')

        print(f'Max closed loop eigenvalue of vertices polytope: {max_eig}')

        max_eig = -np.inf
        for idx in tqdm.tqdm(range(10000)):
            # generate random matrix
            B1_found, B2_found, B3_found = build_B1(np.random.uniform()), build_B2(np.random.uniform()), build_B3(np.random.uniform())
            Bs = [B1_found, B2_found, B3_found]
            for jdx in range(3):
                cl = A + Bs[jdx] @ K
                # print(np.linalg.eigvals(cl))
                if np.max(np.linalg.eigvals(cl).real) > max_eig:
                    max_eig = np.max(np.linalg.eigvals(cl))

                if not all(np.linalg.eigvals(cl).real < 0.):
                    print(f'Found eig > 0 with couple: {A}, {Bs[jdx]}')

        print(f'Max closed loop eigenvalue of vertices polytope: {max_eig}')


