# taken from
# Counter-Example Guided Inductive Synthesis of Control Lyapunov Functions for Uncertain Systems
# by Daniele Masti, Filippo Fabiani, Giorgio Gnecco, and Alberto Bemporad

import cvxpy as cp
import numpy as np
import scipy as sc
import tqdm
from scipy.optimize import direct, Bounds
import matplotlib.pyplot as plt


def build_schur(A, B, P, K):
    return np.block([
        [P, (A + B @ K).T @ P],
        [P.T @ (A + B @ K), P]
    ])

def min_eig(values):
    A = values.reshape((4, 4))
    return np.min(sc.linalg.eigvals(A))

def min_schur_eig(values, *args):
    A = values[:4*4].reshape((4, 4))
    B = values[4*4:].reshape((4, 1))
    P, K = args
    schurri = np.block([
        [P, (A + B @ K).T @ P],
        [P.T @ (A + B @ K), P]
    ])
    return np.min(sc.linalg.eigvals(schurri).real)

def closest_vertex(values, lb, ub):
    # finds the closest vertex of the convex hull given the values
    values_a = values[:4*4]
    values_b = values[4*4:]
    lb_a = lb[0]
    lb_b = lb[1]
    ub_a = ub[0]
    ub_b = ub[1]
    res_a = np.zeros(values_a.shape)
    res_b = np.zeros(values_b.shape)

    # closest vertex of A
    for idx,v in enumerate(values_a):
        # find the closest bound
        dist_lb = (v-lb_a[idx])**2
        dist_ub = (v - ub_a[idx]) ** 2
        if dist_lb < dist_ub:
            res_a[idx] = lb_a[idx]
        else:
            res_a[idx] = ub_a[idx]

    # closest vertex of B
    for idx,v in enumerate(values_b):
        # find the closest bound
        dist_lb = (v-lb_b[idx])**2
        dist_ub = (v - ub_b[idx]) ** 2
        if dist_lb < dist_ub:
            res_b[idx] = lb_b[idx]
        else:
            res_b[idx] = ub_b[idx]

    return res_a.reshape((4,4)), res_b.reshape((4, 1))


# uncertain matrix setting
n = 4
m = 1
epsi = 1e-1
eta = 100.
# discrete time, |eig| < 1
A_min = np.array([
                    [-0.6685, -0.8709, -0.2028, -1.5547],
                    [1.1457, -0.5898, 0.5688, 0.8496],
                    [-0.7812, -0.5754, -0.8774, -0.2501],
                    [-1.1429, 0.1730, 0.7763, 0.1618]
                ])

A_max = np.array([
                    [-0.6295, -0.8202, -0.1910, -1.4641],
                    [1.2166, -0.5555 , 0.6040, 0.9022 ],
                    [-0.7357, -0.5419 , -0.8263, -0.2355 ],
                    [-1.0763, 0.1837, 0.8243, 0.1718],
                ])

delta_a = 0.5*(A_max - A_min)
A_avg = 0.5 * (A_max + A_min)

# in the paper, they only show uncertain A -- here add also uncertain B

# B is dimension (n, m)
B_min = np.array([
    [0.],
    [0.],
    [0.],
    [0.8]
])

B_max = np.array([
    [0.01],
    [0.01],
    [0.01],
    [1.]
])
delta_b = 0.5*(B_max - B_min)
B_avg = 0.5 * (B_max + B_min)

# this is the number of combinations of variable parameters
combinations_a = int(2 ** (n*n))
all_uncertain_A = [A_avg]
for idx in range(combinations_a):
    # every vertex is represented with a binary number
    addition_a = np.multiply(delta_a.reshape(1, -1), np.array([int(b) for b in np.binary_repr(idx, n*n)]))
    # add the vertex to the minimum matrix to get one other vertex
    all_uncertain_A += [(A_min.reshape(1, -1) + addition_a).reshape((n,n))]

combinations_b = int(2 ** (n*m))
all_uncertain_B = [B_avg]
for idx in range(combinations_b):
    # every vertex is represented with a binary number
    addition_b = np.multiply(delta_b.reshape(1, -1), np.array([int(b) for b in np.binary_repr(idx, n*m)]))
    # add the vertex to the minimum matrix to get one other vertex
    all_uncertain_B += [(B_min.reshape(1, -1) + addition_b).reshape((n,m))]



######################
# tentative cegis
######################

found_lyap = False
sampled_A = [A_avg]
sampled_B = [B_avg]
iteration = 0

while not found_lyap:
    print('-' * 80)
    print(f'Iteration {iteration}')
    print('-' * 80)
    iteration += 1

    X = cp.Variable((n,n), symmetric=True)
    W = cp.Variable((m,n))

    uncertain_constraints = []
    for idx in range(len(sampled_A)):
        M = cp.bmat([
            [X, X @ sampled_A[idx].T + W.T @ sampled_B[idx].T],
            [(X @ sampled_A[idx].T + W.T @ sampled_B[idx].T).T, X]
        ])
        uncertain_constraints += [M>>epsi * np.eye(n+n)]

    uncertain_constraints += [X << eta*np.eye(n)]

    prob = cp.Problem(cp.Minimize(0.), uncertain_constraints)
    prob.solve()

    if X.value is None:
        print('Optimisation failed!')
        exit()

    X_star, W_star = X.value, W.value
    P = np.linalg.inv(X_star)
    K = W_star @ P
    print(f'Candidate K: {K}')

    # result from the paper
    # Kbar = np.array([
    #     [1.7667, 0.9014, -0.3555, 1.0089]
    # ])

    # find a cex
    # parameters are values of A and B
    lb_ab = np.hstack([A_min.reshape(1, -1)[0], B_min.reshape(1, -1)[0]])
    ub_ab = np.hstack([A_max.reshape(1, -1)[0], B_max.reshape(1, -1)[0]])
    bounds = Bounds(lb_ab, ub_ab)
    res = direct(min_schur_eig, bounds=bounds, args=(P, K))
    print(f'Min eigenvalue of variable Schur matrix: {res.fun}')
    A_found = res.x[:4 * 4].reshape(4, 4)
    B_found = res.x[4 * 4:].reshape(4, 1)
    print(f'Max eigenvalue of closed loop matrix: {np.max(np.linalg.eigvals(A_found + B_found @ K))}')

    if res.fun <= 0:
        print('found cex')
        A_cex, B_cex = closest_vertex(res.x, [A_min.reshape(1, -1)[0], B_min.reshape(1, -1)[0]],
                                              [A_max.reshape(1,-1)[0], B_max.reshape(1, -1)[0]])
        sampled_A += [A_cex]
        sampled_B += [B_cex]
    else:
        print('valid Lyap')
        found_lyap = True


        # sanity check. compute all possible A+BK, check if eigs are inside the unit circle
        max_eig = 0.
        print('Checking eigenvalues of closed loop matrix from all corners of convex hull...')
        for idx in tqdm.tqdm(range(combinations_a)):
            for jdx in range(combinations_b):
                cl = all_uncertain_A[idx] + all_uncertain_B[jdx] @ K
                # print(np.linalg.eig(cl)[0])
                if np.max(abs(np.linalg.eig(cl)[0])) > max_eig:
                    max_eig = np.max(abs(np.linalg.eig(cl)[0]))

                if not all(abs(np.linalg.eig(cl)[0]) < 1.):
                    print(f'Found eig > 1 with couple: {all_uncertain_A[idx]}, {all_uncertain_B[jdx]}')

        print(f'Max closed loop eigenvalues of vertices polytope: {max_eig}')


