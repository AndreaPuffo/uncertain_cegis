# taken from
# Systematic Synthesis of Passive Fault-Tolerant Augmented Neural
# Lyapunov Control Laws for Nonlinear Systems
# by Grande davide et al.

import cvxpy as cp
import numpy as np
import scipy as sc
import tqdm
from scipy.optimize import direct, Bounds
from utils import get_condition_b_constraint, get_condition_c_constraint, get_condition_a_constraint
from plot_ellipse_matrix_form import plot_ellipse_matrix_form
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

def build_B1(health):
    return np.array([
        [health * np.sin(alpha1) / mass,
             np.sin(alpha2) / mass,
                 np.sin(alpha3) / mass],
        [health * (-np.sin(alpha1)*l1y + np.cos(alpha1)*l1x)/J,
            (-np.sin(alpha2)*l2y + np.cos(alpha2)*l2x)/J,
                (-np.sin(alpha3)*l3y + np.cos(alpha3)*l3x)/J]
    ])

def build_B2(health):
    return np.array([
        [np.sin(alpha1) / mass,
         health * np.sin(alpha2) / mass,
         np.sin(alpha3) / mass],
        [(-np.sin(alpha1) * l1y + np.cos(alpha1) * l1x) / J,
         health * (-np.sin(alpha2) * l2y + np.cos(alpha2) * l2x) / J,
         (-np.sin(alpha3) * l3y + np.cos(alpha3) * l3x) / J]
    ])

def build_B3(health):
    return np.array([
        [np.sin(alpha1) / mass,
         np.sin(alpha2) / mass,
         health * np.sin(alpha3) / mass],
        [ (-np.sin(alpha1) * l1y + np.cos(alpha1) * l1x) / J,
         (-np.sin(alpha2) * l2y + np.cos(alpha2) * l2x) / J,
         health * (-np.sin(alpha3) * l3y + np.cos(alpha3) * l3x) / J]
    ])

def min_hurwi_eig_single(values, *args):
    # build matrices A, B
    values_a = values[:2]
    values_b = values[2:]
    iter, P, K = args

    A = build_A(values_a)
    # build matrix B
    if iter == 1:
        B = build_B1(values_b[0])
    elif iter == 2:
        B = build_B2(values_b[0])
    elif iter == 3:
        B = build_B3(values_b[0])
    else:
        raise ValueError
    # we want (A+BK)P + P(A+BK)^T to have eigs < 0, but change sign for Minimization
    iota = 0.
    hurwi = - (A + B @ K) @ P - P @ (A + B @ K).T - iota * np.eye(A.shape[0])
    return np.min(sc.linalg.eigvals(hurwi).real)


def min_cloop_eig(values, *args):
    # build matrix B
    values_a = values[:2]
    values_b = values[2:]
    A = build_A(values_a)
    # build matrix B
    B1 = build_B1(values_b[0])
    B2 = build_B2(values_b[1])
    B3 = build_B3(values_b[2])
    P, K = args
    # we want (A+BK) to have eigs < 0, but change sign for Minimization
    iota = 0.
    hurwi1 = - (A + B1 @ K)
    hurwi2 = - (A + B2 @ K)
    hurwi3 = - (A + B3 @ K)
    return np.min([
        np.min(sc.linalg.eigvals(hurwi1).real),
        np.min(sc.linalg.eigvals(hurwi2).real),
        np.min(sc.linalg.eigvals(hurwi3).real)])

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
    B1 = build_B1(np.round(values_b[0]))
    B2 = build_B2(np.round(values_b[1]))
    B3 = build_B3(np.round(values_b[2]))

    return A, [B1, B2, B3]


def closest_vertex_single(values, lb, ub, index):
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
    if index == 1:
        B = build_B1(np.round(values_b[0]))
    elif index == 2:
        B = build_B2(np.round(values_b[0]))
    elif index == 3:
        B = build_B3(np.round(values_b[0]))
    else:
        raise ValueError

    return A, B


# uncertain matrix setting
n = 2
m = 3
epsi = 1e-6
eta = 1e-3

domain = [-1., 1.]

A_max = np.array([
                [(-Xu - 2*Xuu * domain[1]) / mass, 0.],
                [0., (-Nr - 2*Nrr*domain[1])/J    ]
])
A_min = np.array([
                [(-Xu - 2*Xuu * domain[0]) / mass, 0.],
                [0., (-Nr - 2*Nrr*domain[0])/J    ]
])
A_avg = 0.5 * (A_max + A_min)

vertex_A = [ A_avg,
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

B_avg1 = 0.5 * (B_max + B_min1)
B_avg2 = 0.5 * (B_max + B_min2)
B_avg3 = 0.5 * (B_max + B_min3)
vertex_B = [B_avg1, B_avg2, B_avg3, B_min1, B_min2, B_min3, B_max]



######################
# tentative cegis
######################

found_lyap = False
sanity_check = True  # if you dont need the sanity check, set to False
sampled_A = [A_avg]
sampled_B = [[B_avg1, B_avg2, B_avg3]]
iteration = 0
domain_ellipse = np.eye(n) * 0.01
fig, ax = plt.subplots(1, 1)
colors = ['b', 'g', 'c', 'y', 'tab:orange', 'r', 'm']


# should converge in 5 iterations
while not found_lyap:
    print('-'*80)
    print(f'Iteration {iteration}')
    print('-' * 80)
    iteration += 1

    # inv_ellipse_P = np.linalg.inv(Ellipse_P)
    # Q = rho * inv_ellipse_P
    Q = cp.Variable((n, n), symmetric=True)
    # H = cp.Variable((m,n))
    # G = H @ Q
    G = cp.Variable((m, n))
    # Y is the new K
    Y = cp.Variable((m, n))

    ### saturation constraints

    # impose ellipse at least contains a ball (domain ellipse)
    gamma_ellipse = cp.Variable((1, 1))
    a = get_condition_a_constraint(gamma=gamma_ellipse, domain_ellipse=domain_ellipse, Q=Q)

    # impose the stability constraints
    bs = []
    for A_mat in sampled_A:
        for Bs in sampled_B:
            for B_mat in Bs:
                bs += get_condition_b_constraint(A_mat, B_mat, Q, G, Y, eta=eta)

    # impose the constraints that the lyapunov ellipse is within L(H)
    c = get_condition_c_constraint(G, Q)

    uncertain_constraints = a + bs + c + [Q >> epsi * np.eye(n)]

    prob = cp.Problem(cp.Minimize(0.), uncertain_constraints)
    prob.solve()

    if Q.value is None:
        print('Optimisation failed!')
        exit()

    # recover values
    # ellipse:
    Q = Q.value
    P_ellipse = np.linalg.inv(Q)
    K = Y.value @ P_ellipse

    plot_ellipse_matrix_form(P_ellipse, ax=ax, edgecolor=colors[(iteration - 1) % len(colors)], label=iteration)

    print(f'Ellipse/Lyapunov P: {P_ellipse}')
    print(f'Candidate K: {K}')

    # find a cex
    # parameters are values of A and B
    lb_ab = [domain[0], domain[0], 0., 0., 0.]
    ub_ab = [domain[1], domain[1], 1., 1., 1.]
    bounds = Bounds(lb_ab, ub_ab)

    # NOTA: this check is also just a sanity check
    res = direct(min_cloop_eig, bounds=bounds, args=(P_ellipse, K), eps=1.)
    print(f'Max eigenvalue of variable closed loop matrix: {-res.fun}')
    A_found, Bs_found = closest_vertex(res.x, lb_ab, ub_ab)
    print(f'Max eigenvalue of closed loop matrix (vertex): '
          f'{np.max(np.linalg.eigvals(A_found + Bs_found[0] @ K))}, '
          f'{np.max(np.linalg.eigvals(A_found + Bs_found[1] @ K))}, '
          f'{np.max(np.linalg.eigvals(A_found + Bs_found[2] @ K))}')

    # split B matrix bounds
    # the disjunction is done here, instead of in the function min_hurwi_eig
    # should be the same, just better looking
    max_eig_so_far = -np.inf
    for idx_b_mat in range(3):
        res = direct(min_hurwi_eig_single, bounds=Bounds(lb_ab[:3], ub_ab[:3]), args=(idx_b_mat + 1, P_ellipse, K), eps=1.)
        print(f'Max eigenvalue of hurwitz lyapunov matrix: {-res.fun}')
        A_found, B_found = closest_vertex_single(res.x, lb_ab[:3], ub_ab[:3], idx_b_mat + 1)
        print(f'Max eigenvalue of closed loop matrix -- found with hurwitz (vertex): '
              f'{np.max(np.linalg.eigvals(A_found + B_found @ K))}')
        if np.max(np.linalg.eigvals(A_found + B_found @ K)) > max_eig_so_far:
            max_eig_so_far = np.max(np.linalg.eigvals(A_found + B_found @ K))
            max_eig_index = idx_b_mat + 1
            max_res = res

    if max_res.fun <= 0:
        print('found cex')
        A_cex, B_cex = closest_vertex_single(res.x, lb_ab, ub_ab, max_eig_index)
        sampled_A += [A_cex]
        sampled_B += [[B_cex]]
    else:
        print('valid Lyap')
        found_lyap = True

        # sanity check. compute all possible A+BK, check if eigs are left side of plane
        max_eig = -np.inf
        print('Checking eigenvalues of closed loop matrix from all corners of convex hull...')
        for idx in tqdm.tqdm(range(len(vertex_A))):
            for jdx in range(len(vertex_B)):
                cl = vertex_A[idx] + vertex_B[jdx] @ K
                # print(np.linalg.eigvals(cl))
                if np.max(np.linalg.eigvals(cl).real) > max_eig:
                    max_eig = np.max(np.linalg.eigvals(cl))

                if not all(np.linalg.eigvals(cl).real < 0.):
                    print(f'Found eig > 0 with couple: \n{vertex_A[idx]}, \n{vertex_B[jdx]}')

        print(f'Max eigenvalues of closed loop: {max_eig}')

        print('Checking random generated matrices...')
        max_eig = -np.inf
        if sanity_check:
            for idx in tqdm.tqdm(range(500000)):
                # generate random matrix
                A_rnd = build_A(
                    [np.random.uniform(low=domain[0], high=domain[1]), np.random.uniform(low=domain[0], high=domain[1])]
                )
                B1_found, B2_found, B3_found = (build_B1(np.random.uniform()),
                                                build_B2(np.random.uniform()), build_B3(np.random.uniform()))
                Bs = [B1_found, B2_found, B3_found]
                for jdx in range(m):
                    cl = A_rnd + Bs[jdx] @ K
                    # print(np.linalg.eigvals(cl))
                    if np.max(np.linalg.eigvals(cl).real) > max_eig:
                        max_eig = np.max(np.linalg.eigvals(cl))

                    if not all(np.linalg.eigvals(cl).real < 0.):
                        print(f'Found eig > 0 with couple: {A_rnd}, {Bs[jdx]}')

            print(f'Max closed loop eigenvalue of random matrices: {max_eig}')


plt.legend()
plt.grid()
plt.show()
