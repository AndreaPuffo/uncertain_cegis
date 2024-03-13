# example taken from
# Counter-Example Guided Inductive Synthesis of Control Lyapunov Functions for Uncertain Systems
# Daniele Masti , Filippo Fabiani , Giorgio Gnecco , and Alberto Bemporad

import cvxpy as cp
import numpy as np
import tqdm
from scipy.optimize import direct, Bounds
import matplotlib.pyplot as plt
from utils import (get_condition_a_constraint, get_condition_b_constraint, get_condition_c_constraint,
                        min_hurwitz_eig, min_closedloop_eig)
from plot_ellipse_matrix_form import plot_ellipse_matrix_form


def min_hurwi_eig_from_values(values, *args):
    A = values[:4*4].reshape((4, 4))
    B = values[4*4:].reshape((4, 1))
    Q, K = args
    # we want (A+BK)P + P(A+BK)^T to have eigs < 0, but change sign for Minimization
    return min_hurwitz_eig(A, B, K, Q)

def min_cloop_eig_from_values(values, *args):
    A = values[:4*4].reshape((4, 4))
    B = values[4*4:].reshape((4, 1))
    P, K = args
    # we want (A+BK) to have eigs < 0, but change sign for Minimization
    return min_closedloop_eig(A, B, K)

def closest_vertex(values, lb, ub):
    # finds the closest vertex of the convex hull given the values
    # because the DiRect algorithm returns *almost* the vertex
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
epsi = 0.00001
eta = -0.001
sanity_check = False
#######################################
# trying Continuous Time setting
# continuous time, Real{eig} < 0
#######################################
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

# tentative uncertain B
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
domain_ellipse = np.eye(n) * 0.01
fig, ax = plt.subplots(1, 1)
colors = ['b', 'g', 'c', 'y', 'tab:orange', 'r', 'm']

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
    gamma = cp.Variable((1, 1))
    a = get_condition_a_constraint(gamma=gamma, domain_ellipse=domain_ellipse, Q=Q)

    # impose the stability constraints
    bs = []
    for A_mat in sampled_A:
        for B_mat in sampled_B:
            bs += get_condition_b_constraint(A_mat, B_mat, Q, G, Y, eta=eta)

    # impose the constraints that the lyapunov ellipse is within L(H)
    c = get_condition_c_constraint(G, Q)

    uncertain_constraints = a + bs + c + [Q>>epsi * np.eye(n)]

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

    plot_ellipse_matrix_form(P_ellipse, ax=ax, edgecolor=colors[(iteration-1) %len(colors)], label=iteration)

    print(f'Ellipse/Lyapunov P: {P_ellipse}')
    print(f'Candidate K: {K}')

    # find a cex
    # parameters are values of A and B
    lb_ab = np.hstack([A_min.reshape(1, -1)[0], B_min.reshape(1, -1)[0]])
    ub_ab = np.hstack([A_max.reshape(1, -1)[0], B_max.reshape(1, -1)[0]])
    bounds = Bounds(lb_ab, ub_ab)

    res = direct(min_cloop_eig_from_values, bounds=bounds, args=(None, K), eps=1.)
    A_found = res.x[:4 * 4].reshape(4, 4)
    B_found = res.x[4 * 4:].reshape(4, 1)
    print(f'Max eigenvalue of closed loop matrix (DiRect): {np.max(np.linalg.eigvals(A_found + B_found @ K))}')
    A_found_vtx, B_found_vtx = closest_vertex(res.x, [A_min.reshape(1, -1)[0], B_min.reshape(1, -1)[0]],
                                              [A_max.reshape(1, -1)[0], B_max.reshape(1, -1)[0]])
    print(f'Max eigenvalue of closed loop matrix (Closest Vtx): '
          f'{np.max(np.linalg.eigvals(A_found_vtx + B_found_vtx @ K))}')

    res = direct(min_hurwi_eig_from_values, bounds=bounds, args=(Q, K))
    print(f'Max eigenvalue of hurwitz lyapunov matrix: {-res.fun}')
    A_found = res.x[:4 * 4].reshape(4, 4)
    B_found = res.x[4 * 4:].reshape(4, 1)
    print(f'Max eigenvalue of hurwitz lyapunov (DiRect): {np.max(np.linalg.eigvals(A_found + B_found @ K))}')
    A_found_vtx, B_found_vtx = closest_vertex(res.x, [A_min.reshape(1, -1)[0], B_min.reshape(1, -1)[0]],
                                              [A_max.reshape(1, -1)[0], B_max.reshape(1, -1)[0]])
    print(f'Max eigenvalue of hurwitz lyapunov (Closest Vtx): '
          f'{np.max(np.linalg.eigvals(A_found_vtx + B_found_vtx @ K))}')

    if res.fun <= 0:
        print('found cex')
        A_cex, B_cex = closest_vertex(res.x, [A_min.reshape(1, -1)[0], B_min.reshape(1, -1)[0]],
                                              [A_max.reshape(1,-1)[0], B_max.reshape(1, -1)[0]])
        sampled_A += [A_cex]
        sampled_B += [B_cex]
    else:
        print('valid Lyap')
        found_lyap = True


        # sanity check. compute all possible A+BK, check if eigs are left side of plane
        max_eig = -np.inf
        if sanity_check:
            print('--- Sanity check ---')
            print('Checking eigenvalues of closed loop matrix from all corners of convex hull...')
            for idx in tqdm.tqdm(range(combinations_a)):
                for jdx in range(combinations_b):
                    cl = all_uncertain_A[idx] + all_uncertain_B[jdx] @ K
                    # print(np.linalg.eigvals(cl))
                    if np.max(np.linalg.eigvals(cl).real) > max_eig:
                        max_eig = np.max(np.linalg.eigvals(cl))

                    if not all(np.linalg.eigvals(cl).real < 0.):
                        print(f'Found eig > 0 with couple: {all_uncertain_A[idx]}, {all_uncertain_B[jdx]}')

            print(f'Max closed loop eigenvalue of vertex polytope: {max_eig}')


plt.legend()
plt.grid()
plt.show()
