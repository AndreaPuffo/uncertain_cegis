# taken from https://stackoverflow.com/questions/56414270/implement-lmi-constraint-with-cvxpy

import cvxpy as cp
import numpy as np
import matplotlib.pyplot as plt

gamma = cp.Variable()

MAT1 = cp.Variable((2, 2))
constraints_2 = [MAT1 >> 0, MAT1[0, 0] == 1, MAT1[1, 0] == 1 + MAT1[0, 1], MAT1[1, 1] == 3]
prob = cp.Problem(cp.Minimize(MAT1[0, 1]), constraints_2)

prob.solve()

print(MAT1[0, 1].value)

# lmi for Lyapunov-like problems
A1 = np.array([
    [-0.9, 0.1],
    [0., -0.8]
])

A2 = np.array([
    [-0.1, 0.3],
    [0., -5.]
])

P = cp.Variable((2, 2), symmetric=True)
constraints = [
    A1 @ P + P @ A1 << 0.1 * np.eye(2),
    A2 @ P + P @ A2 << 0.1 * np.eye(2),
    P >> 0.1 * np.eye(2),
    P[0,1] == 0.
]

prob = cp.Problem(cp.Minimize(cp.trace(P)), constraints)
prob.solve()

print(P.value)

# example 6.3 of LMI in control theory book
A = np.array([
    [0., 1., 0.],
    [1., 1., 0.],
    [-1., 0., 0.]
])

B = np.array([
    [0.],
    [1.],
    [0.]
])

P = cp.Variable((3,3), symmetric=True)
W = cp.Variable((1, 3))

M = cp.bmat([
    [-P, A@P + B@W],
    [P@A.T + W.T @ B.T, -P]
])

prob = cp.Problem(cp.Minimize(0.), [M<<-0.01*np.eye(6), P>>0.01 * np.eye(3), W <= 0.01])
prob.solve()

# matrix ** -1 returns the inversion element-wise
K = W.value @ np.linalg.inv(P.value)
cl = A + B@K
print(f'eigenvalues: {np.linalg.eig(cl)[0]}')

assert all(abs(np.linalg.eig(cl)[0])< 1.)

print(P.value)
print(W.value)


# uncertain matrix setting
n = 2
m = 1
epsi = 0.01
eta = 0.1
power = int(2 ** (n*n))
# discrete time, |eig| < 1
A_min = np.array([
    [0.10, 0.4],
    [0., -0.25]
])
delta_a = 0.5
A_avg = A_min + 0.5 * delta_a * np.ones((n,n))

uncertain_A = []
for idx in range(power):
    # every vertex is represented with a binary number
    addition = delta_a * np.array([int(b) for b in np.binary_repr(idx, n*n)])
    # add the vertex to the minimum matrix to get one other vertex
    uncertain_A += [(A_min.reshape(1, -1) + addition).reshape((n,n))]

# this is the Frobenius norm, which is always >= than the 2-norm
max_matrix_distance = 0.5 * delta_a * n
# B is dimension (n, m)
B_center = np.array([
    [0.],
    [1.]
])

X = cp.Variable((n,n), symmetric=True)
W = cp.Variable((m,n))

uncertain_constraints = []
for idx in range(power):
    M = cp.bmat([
        [X, X @ uncertain_A[idx].T + W.T @ B_center.T],
        [(X @ uncertain_A[idx].T + W.T @ B_center.T).T, X]
    ])

    uncertain_constraints += [M>>epsi * np.eye(n+n)]

uncertain_constraints += [X << eta*np.eye(n)]


print('-'*80)
print('- Uncertain System')
print('-'*80)

prob = cp.Problem(cp.Minimize(0.), uncertain_constraints)
prob.solve()

print(X.value)
print(W.value)

K = W.value @ np.linalg.inv(X.value)
for idx in range(power):
    cl = uncertain_A[idx] + B_center@K
    # print(np.linalg.eig(cl)[0])
    if not all(abs(np.linalg.eig(cl)[0]) < 1.):
        print('wrong')


######################
# tentative cegis
######################

X = cp.Variable((n,n), symmetric=True)
W = cp.Variable((m,n))

uncertain_constraints = []
M = cp.bmat([
    [X, X @ A_avg.T + W.T @ B_center.T],
    [(X @ A_avg.T + W.T @ B_center.T).T, X]
])

uncertain_constraints += [M>>epsi * np.eye(n+n)]

uncertain_constraints += [X << eta*np.eye(n)]

prob = cp.Problem(cp.Minimize(0.), uncertain_constraints)
prob.solve()

X_star, W_star = X.value, W.value
P = np.linalg.inv(X_star)
K = W_star @ P

Schur = np.block([
                    [P, (A_avg + B_center @K).T @ P],
                    [P.T @ (A_avg + B_center @K), P]
                ])

min_schur_eig = np.min(np.linalg.eig(Schur)[0])
if min_schur_eig - max_matrix_distance < 0.:
    print('fregato')

for idx in range(power):
    cl = uncertain_A[idx] + B_center@K
    # print(np.linalg.eig(cl)[0])
    if not all(abs(np.linalg.eig(cl)[0]) < 1.):
        print('wrong')


