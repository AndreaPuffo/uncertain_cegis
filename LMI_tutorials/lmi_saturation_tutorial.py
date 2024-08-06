# taken from https://stackoverflow.com/questions/56414270/implement-lmi-constraint-with-cvxpy

import cvxpy as cp
import numpy as np
import matplotlib.pyplot as plt

# repeatability
SEED = 999
np.random.seed(SEED)

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


P = cp.Variable((2, 2), symmetric=True)
constraints = [
    A1 @ P + P @ A1 << 0.1 * np.eye(2),
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

u_max = 0.15 * np.eye(1)
# saturation condition
M2 = cp.bmat([
    [u_max**2,  W],
    [W.T,       P]
])

num_tolerance = 0.001
prob = cp.Problem(cp.Minimize(0.), [M<<-num_tolerance*np.eye(6), M2>>0.,
                                    P>>num_tolerance * np.eye(3), W <= num_tolerance])
prob.solve()

# matrix ** -1 returns the inversion element-wise
K = W.value @ np.linalg.inv(P.value)
cl = A + B@K
print(f'eigenvalues: {np.linalg.eigvals(cl)}')

assert all(abs(np.linalg.eig(cl)[0])< 1.)

time = np.arange(20.)
x0 = np.random.randn(3,)
x = [x0]
ctrl = [K @ x0]
for t in time:
    x.append((A+B@K) @ x[-1])
    ctrl.append(K @ x[-1])

plt.plot(x, label='States')
plt.plot(ctrl, label='Control')
plt.legend()
plt.show()

print(P.value)
print(W.value)



