import numpy as onp
import matplotlib.pyplot as plt

def simulateController(benchmark, K, horizon, paraSize, variableBounds, label, axes, x0=None):

    stateSize, inputSize = K.shape[1], K.shape[0]

    story = []
    if x0 is None:
        xState = onp.zeros((1, stateSize))
        xState[0, 0] = 0
        xState[0, 1] = 0
    else:
        xState = onp.reshape(x0, (1, stateSize))
    p = onp.ones((1, paraSize))
    for k in range(0, horizon):
        ref = onp.array(xState * 0)
        ref[0, 0] = onp.cos(k / 1000) / 5 + 0.5
        ref[0, 1] = onp.cos((600 + k) / 1000) / 5
        # err=
        # xState[0,-1]+=-.1
        uF = onp.clip((K @ (xState - ref).T).ravel(),
                      variableBounds.lb[stateSize:stateSize + inputSize],
                      variableBounds.ub[stateSize:stateSize + inputSize])
        # uF=onp.array(K@(xState-ref).T).ravel()

        xN = benchmark.innerDynamic(xState, uF, p)
        p = onp.ones((1, paraSize))
        if k >= 3000 and k <= 4000:
            p[0, 2] = .1
        if k >= 5000:
            p[0, 1] = .1
        else:
            pass
        story += [(xState, uF)]
        xState = onp.array(onp.reshape(xN, (1, stateSize)))


    state_trajectory = onp.vstack([x[0] for x in story])
    input_trajectory = onp.vstack([x[1] for x in story])

    axes[0].title.set_text('Input ' + str(label))
    axes[0].plot(input_trajectory)
    axes[0].set_xlabel('Time Steps')
    axes[0].grid()

    axes[1].title.set_text('State ' + str(label))
    axes[1].plot(state_trajectory[:, :2])
    axes[1].set_xlabel('Time Steps')
    axes[1].grid()


def plotEllipse(Q, colorString, label):

    # Q=np.linalg.inv(Psat)
    # Create a grid of points
    x = onp.linspace(-2, 2, 100)
    y = onp.linspace(-2, 2, 100)
    X, Y = onp.meshgrid(x, y)
    Z = onp.zeros_like(X)

    # Compute the quadratic form values on the grid
    for i in range(X.shape[0]):
        for j in range(X.shape[1]):
            vec = onp.array([X[i, j], Y[i, j]]).reshape((2, 1))
            Z[i, j] = vec.T @ Q @ vec

    contour = plt.contour(X, Y, Z, levels=[1], colors=colorString)
    plt.xlabel('x1')
    plt.ylabel('x2')
    plt.title('Contour plot of $x^T Q x = 1$')
    # plt.gca().set_aspect('equal', adjustable='box')
    plt.clabel(contour, contour.levels, fmt=lambda x: str(label))