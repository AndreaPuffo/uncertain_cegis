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
# x1dot =  (-Xu*x1-Xuu*x1**2 + F1x*h1                + F2x*h2                + F3x*h3                  )/m ,
# x2dot = (-Nr*x2-Nrr*x2**2 + (-F1x*l1y+F1y*l1x)*h1 + (-F2x*l2y+F2y*l2x)*h2 + (-F3x*l3y+F3y*l3x)*h3 ) /Jz ],
# where
#     F1x = u1*np.sin(torch.tensor(alpha1))
#     F1y = u1*np.cos(torch.tensor(alpha1))
#     F2x = u2*np.sin(torch.tensor(alpha2))
#     F2y = u2*np.cos(torch.tensor(alpha2))
#     F3x = u3*np.sin(torch.tensor(alpha3))
#     F3y = u3*np.cos(torch.tensor(alpha3))
# Jacobian: df/dx, df/du
# (-Xu - 2Xuu * x1)/ mass, 0. ;
# 0., (-Nr - 2*Nrr*x2)/J   ;
# Jacobian: df/du
# sin(alpha1)/mass, sin(alpha2)/mass, sin(alpha3)/mass
# (-sin(alpha1)*l1y + cos(alpha1)*l1x)/J,  (-sin(alpha2)*l2y + cos(alpha2)*l2x)/J, (-sin(alpha3)*l3y + cos(alpha3)*l3x)/J,

# system parameters
mass = 500.0  #
J = 300.
Xu = 6.106  #
Xuu =  5.0  #
Nr =  210.0  #
Nrr =  3.0  #
l1x =  -1.01
l1y = -0.353
alpha1 = np.deg2rad(110.0)
l2x = -1.01
l2y =  0.353
alpha2 = np.deg2rad(70.0)
l3x = 0.75
l3y =  0.0
alpha3 = np.deg2rad(180.0)


def build_hurwitz(A, B, P, K):
    return (A + B @ K) @ P + P @ (A + B @ K).T

def build_A(values):
    return np.array([
        [(-Xu - 2 * Xuu * values[0]) / mass, 0.],
        [0., (-Nr - 2 * Nrr * values[1]) / J]
    ])

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
        [health1 * np.sin(alpha1) / mass,
            health2 * np.sin(alpha2) / mass,
                health3 * np.sin(alpha3) / mass],
        [health1 * (-np.sin(alpha1)*l1y + np.cos(alpha1)*l1x)/J,
            health2 * (-np.sin(alpha2)*l2y + np.cos(alpha2)*l2x)/J,
                health3 * (-np.sin(alpha3)*l3y + np.cos(alpha3)*l3x)/J]
    ])

def min_hurwi_eig(values, *args):
    # build matrix B
    values_a = values[:2]
    values_b = values[2:]
    A = build_A(values_a)
    B = build_B(values_b[0])
    P, K = args
    # we want (A+BK)P + P(A+BK)^T to have eigs < 0, but change sign for Minimization
    iota = 0.
    hurwi = - (A + B @ K) @ P - P @ (A + B @ K).T - iota * np.eye(A.shape[0])
    return np.min(sc.linalg.eigvals(hurwi).real)

def closest_vertex(values, lb, ub):
    # finds the closest vertex of the convex hull given the values
    values_a = values[:2]
    values_b = values[2:]
    lb_a, lb_b = lb[:2], lb[2:]
    ub_a, ub_b = ub[:2], ub[2:]
    res_a = np.zeros(values_a.shape)
    # closest vertex of B
    for idx,v in enumerate(values_a):
        # find the closest bound
        dist_lb = (v - lb_a[idx])**2
        dist_ub = (v - ub_a[idx]) ** 2
        if dist_lb < dist_ub:
            res_a[idx] = lb[idx]
        else:
            res_a[idx] = ub[idx]

    A = build_A(res_a)
    B = build_B(values_b[0])

    return A, B


# uncertain matrix setting
n = 2
m = 3
epsi = 1e-3
eta = 10.

domain = [-10., 10.]

A_max = np.array([
                [(-Xu - 2*Xuu * domain[1]) / mass, 0.],
                [0., (-Nr - 2*Nrr*domain[1])/J    ]
])
A_min = np.array([
                [(-Xu - 2*Xuu * domain[0]) / mass, 0.],
                [0., (-Nr - 2*Nrr*domain[0])/J    ]
])
A_avg = 0.5 * (A_max + A_min)

all_uncertain_A = [ A_avg,
                    build_A([domain[0], domain[0]]), build_A([domain[0], domain[1]]),
                    build_A([domain[1], domain[0]]), build_A([domain[0], domain[1]])]


# B is dimension (n, m)
B_max = np.array([
    [np.sin(alpha1) / mass, np.sin(alpha2) / mass, np.sin(alpha3) / mass],
    [(-np.sin(alpha1)*l1y + np.cos(alpha1)*l1x)/J,  (-np.sin(alpha2)*l2y + np.cos(alpha2)*l2x)/J, (-np.sin(alpha3)*l3y + np.cos(alpha3)*l3x)/J]
])

B_min1 = np.array([
    [0., np.sin(alpha2) / mass, np.sin(alpha3) / mass],
    [0.,  (-np.sin(alpha2)*l2y + np.cos(alpha2)*l2x)/J, (-np.sin(alpha3)*l3y + np.cos(alpha3)*l3x)/J]
])

B_min2 = np.array([
    [np.sin(alpha1) / mass, 0., np.sin(alpha3) / mass],
    [(-np.sin(alpha1)*l1y + np.cos(alpha1)*l1x)/J,  0., (-np.sin(alpha3)*l3y + np.cos(alpha3)*l3x)/J]
])

B_min3 = np.array([
    [np.sin(alpha1) / mass, np.sin(alpha2) / mass, 0.],
    [(-np.sin(alpha1)*l1y + np.cos(alpha1)*l1x)/J,  (-np.sin(alpha2)*l2y + np.cos(alpha2)*l2x)/J, 0.]
])

B_avg = 0.5 * B_max
all_uncertain_B = [B_avg, B_min1, B_min2, B_min3, B_max]


######################
# tentative cegis
######################

found_lyap = False
sampled_A = [A_avg]
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
        M = P @ sampled_A[idx].T + sampled_A[idx] @ P + W.T @ sampled_B[idx].T + sampled_B[idx] @ W
        uncertain_constraints += [M << -epsi * np.eye(n)]

    uncertain_constraints += [P << eta*np.eye(n), P>> 0.]

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
    lb_ab = [domain[0], domain[0], 0.]
    ub_ab = [domain[1], domain[1], float(m)]
    bounds = Bounds(lb_ab, ub_ab)
    res = direct(min_hurwi_eig, bounds=bounds, args=(P_star, K), maxiter=5000)
    print(f'Max eigenvalue of variable hurwitz matrix: {-res.fun}')
    A_found, B_found = closest_vertex(res.x, lb_ab, ub_ab)
    print(f'Max eigenvalue of closed loop matrix (vertex): {np.max(np.linalg.eigvals(A_found + B_found @ K))}')

    if res.fun <= 0:
        print('found cex')
        A_cex, B_cex = closest_vertex(res.x, lb_ab, ub_ab)
        sampled_A += [A_cex]
        sampled_B += [B_cex]
    else:
        print('valid Lyap')
        found_lyap = True

        # sanity check. compute all possible A+BK, check if eigs are left side of plane
        max_eig = -np.inf
        print('Checking eigenvalues of closed loop matrix from all corners of convex hull...')
        for idx in tqdm.tqdm(range(len(all_uncertain_A))):
            for jdx in range(len(all_uncertain_B)):
                cl = all_uncertain_A[idx] + all_uncertain_B[jdx] @ K
                # print(np.linalg.eigvals(cl))
                if np.max(np.linalg.eigvals(cl).real) > max_eig:
                    max_eig = np.max(np.linalg.eigvals(cl))

                if not all(np.linalg.eigvals(cl).real < 0.):
                    print(f'Found eig > 0 with couple: \n{all_uncertain_A[idx]}, \n{all_uncertain_B[jdx]}')

        print(f'Max eigenvalues of closed loop: {max_eig}')

        print('Checking random generated matrices...')
        max_eig = -np.inf
        for idx in tqdm.tqdm(range(500000)):
            # generate random matrix
            A_rnd = build_A(
                [np.random.uniform(low=domain[0], high=domain[1]), np.random.uniform(low=domain[0], high=domain[1])]
            )
            B_rnd = build_B(np.random.uniform(low=0., high=3.))
            cl = A_rnd + B_rnd @ K
            # print(np.linalg.eigvals(cl))
            if np.max(np.linalg.eigvals(cl).real) > max_eig:
                max_eig = np.max(np.linalg.eigvals(cl))

            if not all(np.linalg.eigvals(cl).real < 0.):
                print(f'Found eig > 0 with couple: {A_rnd}, {B_rnd}')

        print(f'Max closed loop eigenvalue of random matrices: {max_eig}')


