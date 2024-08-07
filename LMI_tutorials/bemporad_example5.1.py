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


EXAMPLE 5.1
"""

n, m = 2, 1
s = 1  # NOTA: example 1 has *fixed* matrix
combinations = int(2.**m)

# differential inclusions for the saturation: need the "corners" matrices
Es = []
for j in range(combinations):
    Es += [np.diag([int(b) for b in np.binary_repr(j, m)])]

A = np.array([
    [1., 1.],
    [0., 1.],
])
As = [A]

B = np.array([
    [0.5],
    [1.],
])
Bs = [B]

# NOTA: in the paper, this matrix is denoted D -- for more intuition, re-defined as Bw
Bw = 0.1 * np.eye(2)
Bws = [Bw]

# domains
u_max = 1.
L = np.array([[0., 0.1]])   # x2_max = 10.
h = L.shape[0]
norm_w = 1.  # w^T * w <= 1

"""
not very clear from the paper, but maybe Ka and Kb are given (?)
"""

Ka = np.array([[-0.3175, -1.1664]])
# Kb = np.array([[-0.0527, -1.0280]])
fig, ax = plt.subplots(1, 1, figsize=(8, 4.5))
fig_max, ax_max = plt.subplots(1, 1, figsize=(8, 4.5))

# line search for tau
taus = [1e-4, 1e-3, 1e-2, 0.1, 0.2, 0.3, 0.4, 0.425, 0.426, 0.45, 0.475,
        0.49, 0.5,  0.6, 0.7, 0.75, 0.775, 0.776, 0.78,
        0.8, 0.9, 1.-1e-2, 1.-1e-3, 1.-1e-4]


for idx in range(len(taus)):

    Q, Y, Z, constraints = constraints_lemma23(As=As, Bs=Bs, Bws=Bws, Es=Es, L=L,
                                               umax=u_max, tau=taus[idx], K=Ka)

    ############################
    # Program solution
    ############################

    # section 4.2 says "the associated invariant set E(Qa) can be computed by solving the SDP
    # in Lemma 2 with objective given as max{trace(Q )} instead of min{trace(Q )}"
    prob = cp.Problem(cp.Maximize( cp.trace(Q) ), constraints=constraints)
    prob.solve(solver='CLARABEL')  #pip install clarabel
    if prob.status == 'optimal':
        Qa = Q.value
        K = Y.value @ np.linalg.inv(Qa)  # this should be Ka or Kb again
        H = Z.value @ np.linalg.inv(Qa)
        print('-'*80)
        print(f' Found a solution for tau: {taus[idx]}')
        print(f'Qa: {Qa}')
        print(f'K: {K}')
        plot_ellipse_matrix_form(np.linalg.inv(Qa), ax, radius_ellipse=1.,
                                 edgecolor=color_cycle[idx%len(color_cycle)], label=taus[idx])

    ######################################################
    # try Lemma 3 and maximisation of region of attraction
    ######################################################
    Qmax, Ymax, Zmax, constraints_max = constraints_lemma23(As=As, Bs=Bs, Bws=Bws, Es=Es, L=L,
                                           umax=u_max, tau=taus[idx], K=None)
    prob_max = cp.Problem(cp.Maximize(cp.trace(Qmax)), constraints=constraints_max)
    try:
        prob_max.solve(solver='CLARABEL')
    except cp.SolverError:
        prob_max.solve()

    if prob_max.status == 'optimal':
        Qmax = Qmax.value
        Kmax = Ymax.value @ np.linalg.inv(Qmax)
        Hmax = Zmax.value @ np.linalg.inv(Qmax)
        print('-'*80)
        print(f' Found a solution for Maximisation problem with tau: {taus[idx]}')
        print(f'QM: {Qmax}')
        print(f'Kb: {Kmax}')
        plot_ellipse_matrix_form(np.linalg.inv(Qmax), ax_max, radius_ellipse=1.,
                                 edgecolor=color_cycle[idx % len(color_cycle)], label=taus[idx])



# plt.xlim([-20, 20])
# plt.ylim([-20, 20])
plt.grid()


x0 = np.array([[-134., 7.]]).T
xs = [x0]

# sanity check for plots:
# this ellipse should approx go from -130 to 130 for x1, and -10 to 10 for x2
plot_ellipse_matrix_form(np.linalg.inv(np.array([
    [1.8024 * 1e4, -0.095*1e4],
    [-0.095*1e4, 0.01*1e4]
])), ax_max, radius_ellipse=1., edgecolor='red', linestyle='dashed', label='Bemporad QM')
plot_ellipse_matrix_form(np.linalg.inv(np.array([
    [393.525, -93.8734],
    [-93.8734, 33.6350]
])), ax, radius_ellipse=1., edgecolor='red', linestyle='dashed', label='Bemporad Qa')
fig.legend()
fig_max.legend()


plt.show()

