# taken from Interpolation based predictive control by ellipsoidal invariant sets
# Daniel Rubin, Pedro Mercader, Per-Olof Gutman, Hoai-Nam Nguyen,
# Alberto Bemporad

import cvxpy as cp
import numpy as np
import matplotlib.pyplot as plt
from bemporad_lemmas import constraints_lemma23
from src.plot_ellipse_matrix_form import plot_ellipse_matrix_form
# from src.plot_ellipse_matrix_form import plot_ellipse_matrix_form

# repeatability
SEED = 999
np.random.seed(SEED)


color_cycle = ['tab:blue', 'tab:orange', 'tab:green', 'tab:red', 'tab:purple', 'tab:brown',
               'tab:pink', 'tab:gray', 'tab:olive', 'tab:cyan']

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


EXAMPLE 5.2
"""

n, m = 4, 1
s = 2  # NOTA: example 1 has *fixed* matrix
combinations = int(2.**m)

# differential inclusions for the saturation: need the "corners" matrices
Es = []
for j in range(combinations):
    Es += [np.diag([int(b) for b in np.binary_repr(j, m)])]

def parametricA(k):
    return np.array([
                            [1., 0., 0.1, 0.],
                            [0., 1., 0., 0.1],
                            [-0.1*k, 0.1*k, 1., 0.],
                            [0.1*k, -0.1*k, 0., 1.]
                    ])
As = [parametricA(k=0.5), parametricA(k=1.5)]

B = np.array([
    [0.],
    [0.],
    [1.],
    [0.]
])
Bs = [B, B]

# NOTA: in the paper, this matrix is denoted D -- for more intuition, re-defined as Bw
Bw = 0. * np.eye(n)
Bws = [Bw, Bw]

# domains
u_max = 1.
L = np.array([
    [0.5, 0., 0., 0.],
    [0., 0.5, 0., 0.],
    [0., 0., 0.1, 0.],
    [0., 0., 0., 0.1]
              ])   # x1, x2 max = 2. /// x3, x4 max = 10
norm_w = 1.  # w^T * w <= 1

"""
not very clear from he paper, but maybe Ka and Kb are given (?)
"""

Ka = np.array([[-13.4535, 7.6441, -6.1226, 5.7299]])

taus = [1e-4, 1e-3, 1e-2, 0.1, 0.2, 0.4, 0.5, 0.6, 0.8, 0.9, 1.-1e-2, 1.-1e-3, 1.-1e-4]

for idx in range(len(taus)):

    Q, Y, Z, constraints = constraints_lemma23(As=As, Bs=Bs, Bws=Bws, Es=Es, L=L,
                                               umax=u_max, tau=taus[idx], K=Ka)

    ############################
    # Program solution
    ############################

    # section 4.2 says "the associated invariant set E(Qa) can be computed by solving the SDP
    # in Lemma 2 with objective given as max{trace(Q )} instead of min{trace(Q )}"
    prob = cp.Problem(cp.Maximize( cp.trace(Q) ), constraints=constraints)
    try:
        prob.solve(solver='CLARABEL')  # pip install clarabel
    except cp.SolverError:
        prob.solve()

    if prob.status == 'optimal':
        Qa = Q.value
        K = Y.value @ np.linalg.inv(Qa)  # this should be Ka or Kb again
        H = Z.value @ np.linalg.inv(Qa)
        print(f' Found a solution for tau: {taus[idx]}')
        print(f'Qa: {Qa}')
        print(f'K: {K}')
