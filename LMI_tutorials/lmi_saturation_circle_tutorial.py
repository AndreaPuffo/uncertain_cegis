# taken from https://stackoverflow.com/questions/56414270/implement-lmi-constraint-with-cvxpy

import cvxpy as cp
import numpy as np
import matplotlib.pyplot as plt
from plot_ellipse_matrix_form import plot_ellipse_matrix_form
from utils import solve_saturation_ctrl_synthesis

# repeatability
SEED = 999
np.random.seed(SEED)




#########
# this follows the approach on Theo 6 in
# Control under Quantization, Saturation and Delay: An LMI Approach
# Emilia Fridman, Michel Dambrine
# actually from
# An analysis and design method for linear systems subject to
# actuator saturation and disturbance
# Tingshu Hua , Zongli Lina , Ben M. Chen

# example 6.3 of LMI in control theory book -- simplified

n, m = 2, 3

A = np.array([
    [0., 1.],
    [1., 1.],
])

B = np.array([
    [0., 1., 0.],
    [1., 0., 2.],
])

u_max = 2.
# the theory saturates the inputs at 1, so we multiply B times the actual saturated value
B = u_max * B

K = np.array([[-5., -5.], [-3., -3.], [-10., -10.]])

print(f'Eigenvalues of closed loop: {np.linalg.eigvals(A + B@K)}')

# input_sat_poly = pt.Polytope(A=np.vstack([K, -K, np.eye(n)]),
#                              b=np.vstack([u_max * np.ones((2*m, 1)), 10.*np.ones((n, 1))]) )
# fig, ax = plt.subplots()
# plt.xlim([-3, 3])
# plt.ylim([-3, 3])
# input_sat_poly.plot(ax=ax)


K1 = cp.Variable((m, m), diag=True)
K2 = cp.Variable((m, m), diag=True)
Ellipse_P = np.eye(n)
M = (A+B@ K1 @K).T @ Ellipse_P + Ellipse_P @ (A+B@ K1 @K) + 0.5*(K.T @ K2 + Ellipse_P@B)@(K2 @ K + B.T @ Ellipse_P)


num_tolerance = 0.001
# prob = cp.Problem(cp.Minimize(0.), [M<<0., K1>>0., K1 << np.eye(n), K1 + K2 >> np.eye(n)])
# prob.solve()

############################
# Sec 2.3 improved
############################

K = cp.Variable((m,n))
prob = cp.Problem(cp.Minimize(0.), [(A +B @ K).T + (A+B@K) << -0.000001 * np.eye(n)])
prob.solve()

K = K.value

combinations = int(2.**m)
H = cp.Variable((m,n))

constraints_improved = []
for c in range(combinations):
    binary_repr = np.binary_repr(c, m)
    rows = []
    for r in range(m):
        if binary_repr[r] == '0':
            rows.append(H[r,:])
        elif binary_repr[r] == '1':
            rows.append(K[r,:])
        else:
            raise ValueError
    feedback_matrix = cp.vstack(rows)
    constraints_improved.append(
        (A + B @ feedback_matrix).T @ Ellipse_P + Ellipse_P @ (A + B @ feedback_matrix) << 0.
    )

prob = cp.Problem(cp.Minimize(0.), constraints_improved)
prob.solve()

H = H.value

###############################
# Sec 2.4 estimation of domain of attraction
###############################

def estimation_domain(domain_ellipse, A, B, K):

    n = A.shape[0]
    m = B.shape[1]
    # inv_ellipse_P = np.linalg.inv(Ellipse_P)
    # Q = rho * inv_ellipse_P
    Q = cp.Variable((n,n), symmetric=True)
    # H = cp.Variable((m,n))
    # G = H @ Q
    G = cp.Variable((m,n))

    # condition a) >> 0
    gamma = cp.Variable((1,1))
    condition_a = [cp.bmat([
        [gamma * domain_ellipse, np.eye(n)],
        [np.eye(n), Q]
    ]) >> 0.]

    KQ = K @ Q
    # condition b << 0
    combinations = int(2**m)
    condition_b = []
    for c in range(combinations):
        binary_repr = np.binary_repr(c, m)
        rows = []
        for r in range(m):
            if binary_repr[r] == '0':
                rows.append(G[r,:])
            elif binary_repr[r] == '1':
                rows.append(KQ[r,:])
            else:
                raise ValueError
        feedback_matrix = cp.vstack(rows)
        condition_b.append(
            # rho * (A + B @ feedback_matrix).T @ inv_ellipse_P +
            # rho * inv_ellipse_P @ (A + B @ feedback_matrix) << 0.
            Q @ A.T + A @ Q + feedback_matrix.T @ B.T + B@feedback_matrix << 0.
        )

    # condition c) >> 0
    condition_c = []
    for i in range(m):
        condition_c.append(
            cp.bmat([
            [np.eye(1),         G[i,:][:,None].T],
            [G[i,:][:,None],    Q]
        ]) >> 0.
        )

    prob = cp.Problem(cp.Minimize(gamma), condition_a + condition_b + condition_c + [Q >> 0.])
    prob.solve()

    return gamma.value, Q.value, G.value

rho_ellipse_domain = 1.3
domain_ellipse = np.eye(n) / rho_ellipse_domain
gamma, Q, G = estimation_domain(domain_ellipse=np.eye(n), A=A, B=B, K=K)

print(f'gamma: {gamma}')

###############################
# Sec 2.5 Controller design
###############################

print('--- Controller design ---')



rho_ellipse_domain = 1.3
domain_ellipse = np.eye(n) / rho_ellipse_domain
gamma, Y, Q, G = solve_saturation_ctrl_synthesis(domain_ellipse=domain_ellipse, A=A, B=B)

# recover values
K = Y @ np.linalg.inv(Q)

# ellipse:
P_ellipse = np.linalg.inv(Q) * rho_ellipse_domain

print(f'alpha: {1./(gamma)**0.5}')
print(f'K: {K}')
print(f'Ellipse P: {P_ellipse}')

fig, ax = plt.subplots(1, 1, figsize=(8, 4.5))
# plt.xlim([-20, 20])
# plt.ylim([-20, 20])
plt.grid()

##############################
# tentative example 1 of the paper
##############################

A = np.array([
    [0., 1.],
    [1., 0.]
])

B = np.array([
    [0.],
    [5.]
])

K = np.array([[-2., -1.]])

rho_domain = 2.
ellipse_domain = np.eye(n) * rho_domain

gamma, Q, G = estimation_domain(domain_ellipse=ellipse_domain, A=A, B=B, K=K)
# Q = rho * P^{-1}
P = np.linalg.inv(Q) * rho_domain

print(f'P: {P}')
print(f'alpha: {1./gamma**0.5}')

plot_ellipse_matrix_form(P, ax, edgecolor='green')

paper_P = np.array([
    [0.1170, 0.0627],
    [0.0627, 0.0558]
])
plot_ellipse_matrix_form(paper_P, ax, edgecolor='orange')


# control synthesis
gamma, Y, Q, G = solve_saturation_ctrl_synthesis(domain_ellipse=ellipse_domain, A=A, B=B)
# recover values
P_ctrl = np.linalg.inv(Q) * rho_domain
plot_ellipse_matrix_form(P_ctrl, ax, edgecolor='red')
alpha = 1./gamma**0.5
print(f'alpha: {alpha}')

plot_ellipse_matrix_form(ellipse_domain/alpha**2, ax, radius_ellipse=1.,
                         edgecolor='black', linestyle='dashed')
plot_ellipse_matrix_form(ellipse_domain, ax, radius_ellipse=alpha**2,
                         edgecolor='red', linestyle='dashed')



K_ctrl = Y @ np.linalg.inv(Q)

x0 = np.array([[15., -15.]]).T
xs = [x0]
sat_xs = [x0]
tau = 0.001
for t in range(2000):
    # continuous time discretized
    xs += [xs[-1] + tau * (A + B@K_ctrl) @ xs[-1]]
    sat_xs += [sat_xs[-1] + tau * (A @ sat_xs[-1] + B @ np.clip(K_ctrl @ sat_xs[-1], -1, 1))]


xs = np.hstack(xs)
sat_xs = np.hstack(sat_xs)

plt.scatter(xs[0,:], xs[1,:])
plt.scatter(sat_xs[0,:], sat_xs[1,:], c='g')

plt.figure()
plt.plot(xs.T)
plt.plot(sat_xs.T)

plt.plot((K_ctrl @ xs)[0], c='r')
plt.plot(np.clip(K_ctrl @ sat_xs, -1, 1)[0], c='g')

plt.grid()

plt.show()
