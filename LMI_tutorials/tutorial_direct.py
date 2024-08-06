import scipy as sc
import numpy as np
from scipy.optimize import direct, Bounds

def styblinski_tang(pos):

    x, y = pos

    return 0.5 * (x**4 - 16*x**2 + 5*x + y**4 - 16*y**2 + 5*y)

bounds = Bounds([-4., -4.], [4., 4.])

result = direct(styblinski_tang, bounds)

print(result.x, result.fun, result.nfev)
# array([-2.90321597, -2.90321597]), -78.3323279095383, 2011

bounds = Bounds([0.5, 0.5, 0.5, 0.5], [1., 1., 1., 1.])
def min_eig(values):
    A = values.reshape((2,2))
    return np.min(sc.linalg.eigvals(A))

result = direct(min_eig, bounds, eps=1e-3)

print(result.x, result.fun)

print(np.linalg.eigvals(
    np.array([
        [0.5, 1.],
        [1., 0.5]
    ])
))
