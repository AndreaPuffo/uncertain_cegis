import jax
jax.config.update("jax_enable_x64", True)
import sys
sys.dont_write_bytecode=True


import jax

from jax import numpy as jnp
from functools import partial
import scipy

import numpy.matlib
import numpy as onp
import matplotlib.pyplot as plt
import jax.experimental

class BaseBenchmark:
    def __init__(self, noiseCov=None, enableNoise=True):
        self.enableNoise = enableNoise
        if noiseCov is None:
            self.noiseCov = 0.1
        else:
            self.noiseCov = noiseCov
            print('overwritten noise cov:', noiseCov)
        pass

    def innerDynamic(self, x0, uT, p):
        raise NotImplementedError


# %%

class quasiLPV(BaseBenchmark):
    def __init__(self, enableNoise):
        super().__init__(0.02, enableNoise)

        global stateSize
        assert stateSize == 3
        global inputSize
        assert inputSize == 2
        self.norm = None
        self.x0 = onp.array([0, 0, 0]).reshape((3, 1))
        self.A = onp.array([[-0.29181893, -0.05231634, 0.],
                            [0.11910899, -1.48769104, -0.43707303],
                            [-0.5, 0.56982177, 0.4038157]]) / 1.5
        self.B = onp.array([[-1.31637896, 1.69459169],
                            [0.37108047, 0.3295812],
                            [1.30842492, -1.04712173]])

        pass

    @partial(jax.jit, static_argnums=(0,))
    def innerDynamic(self, xT, uT, p):
        uT = uT.reshape((inputSize, 1))
        xT = xT.reshape((stateSize, 1))
        xN = jnp.zeros((stateSize, 1))
        xN = self.A @ xT + jnp.tanh(xT) + self.B @ uT + jnp.maximum(self.B @ uT, 0)
        return xN.reshape((stateSize, 1))


class TxT(BaseBenchmark):
    def __init__(self, enableNoise):
        super().__init__(0.02, enableNoise)

        global stateSize
        assert stateSize == 1
        global inputSize
        assert inputSize == 1
        self.norm = None
        self.x0 = onp.array([0]).reshape((1, 1))
        self.A = onp.array([[.69181893]])
        self.B = onp.array([[-1.31637896]])

        pass

    @partial(jax.jit, static_argnums=(0,))
    def innerDynamic(self, xT, uT, p):
        uT = uT.reshape((inputSize, 1))
        xN = jnp.zeros((stateSize, 1))
        xN = self.A @ xT + jnp.tanh(xT) + self.B @ uT + jnp.maximum(self.B @ uT, 0) - 1.55 * jnp.maximum(
            -self.B @ (uT + 0.5), 0)
        # xN=jnp.maximum(xN,xN*0);

        return xN.reshape((stateSize, 1))


class glider(BaseBenchmark):
    def __init__(self, enableNoise):
        super().__init__(0.02, enableNoise)

        global stateSize
        assert stateSize == 2
        global inputSize
        assert inputSize == 2
        global paraSize
        assert paraSize == 1

        pass

    @partial(jax.jit, static_argnums=(0,))
    def innerDynamic(self, xT, uT, p):
        ms = 44.9
        mp = 11
        m1 = 64.84
        m3 = 99.43
        KD0 = 0.04
        KD = 9.44
        KL0 = 2.16
        KL = 4.88
        NablaH = 0.054
        rho = 1027.5
        KFD = 10
        KUDelta = 1
        alpha = (-3.5 + p[0]) / 360 * onp.pi * 2
        theta = (23.7 + p[0]) / 360.0 * onp.pi * 2
        uT = uT.reshape((inputSize))
        xT = xT.reshape((stateSize))
        Vq = jnp.sum(xT ** 2)
        D = (KD0 + KD * alpha ** 2) * Vq
        L = (KL0 + KL * alpha) * Vq
        # B=rho*9.81*(NablaH+uT[0])
        B = uT[0] * 1
        # Fdelta1=KFD*KUDelta*Vq*uT[1]
        # Fdelta2=KFD*KUDelta*Vq*uT[2]
        Fdelta1 = uT[1]
        Fdelta2 = uT[2]
        v1dot = 1 / m1 * (-D * jnp.cos(alpha) + L * jnp.sin(alpha) + jnp.sin(theta) * (B - 9.81 * ms - 9.81 * mp))
        v2dot = 1 / m3 * (-D * jnp.sin(alpha) - L * jnp.cos(alpha) + jnp.cos(theta) * (
                    -B + 9.81 * ms + 9.81 * mp) + Fdelta1 + Fdelta2)
        Ts = 0.125
        xN = jnp.zeros((stateSize, 1))
        xN = xN.at[0].set(xT[0] + Ts * v1dot)
        xN = xN.at[1].set(xT[1] + Ts * v2dot)
        return xN.reshape((stateSize, 1))


class twoStateAUV(BaseBenchmark):
    def __init__(self, enableNoise):
        super().__init__(0.02, enableNoise)

        global stateSize
        assert stateSize == 2
        global inputSize
        assert inputSize == 3
        global paraSize
        assert paraSize == 3

        pass

    @partial(jax.jit, static_argnums=(0,))
    def innerDynamic(self, xT, uT, para):
        xT = xT.reshape((stateSize, 1))
        uT = uT.reshape((inputSize))
        para = para.reshape((paraSize))
        m = 500.0
        Jz = 300.0
        Xu = 6.106
        Xuu = 5.0
        Nr = 210.0
        Nrr = 3.0
        l1x = 1.01
        l1y = -0.353
        alpha1 = 1.9198621771937625
        l2x = -1.01
        l2y = 0.353
        alpha2 = 1.2217304763960306
        l3x = 0.75
        l3y = 0.0
        alpha3 = 3.141592653589793
        h1 = para[0]
        h2 = para[1]
        h3 = para[2]

        pG = 1  # 40 # actuator range here so thatr sat bound is -1/+1
        F1_x = jnp.sin(alpha1) * uT[0] * pG
        F2_x = jnp.sin(alpha2) * uT[1] * pG
        F3_x = jnp.sin(alpha3) * uT[2] * pG

        F1_y = jnp.cos(alpha1) * uT[0] * pG
        F2_y = jnp.cos(alpha2) * uT[1] * pG
        F3_y = jnp.cos(alpha3) * uT[2] * pG

        x1dot = 1 / m * (-Xu * xT[0] - Xuu * xT[0] ** 2 + h1 * F1_x + F2_x * h2 + F3_x * h3)

        x2dot = 1 / Jz * (-Nr * xT[1] - Nrr * xT[1] ** 2 + h1 * (-F1_x * l1y + F1_y * l1x) +
                          h2 * (-F2_x * l2y + F2_y * l2x) + h3 * (-F3_x * l3y + F3_y * l3x))

        Ts = .01
        xN = jnp.zeros((stateSize, 1))
        xN = xN.at[0].set(xT[0] + Ts * x1dot)
        xN = xN.at[1].set(xT[1] + Ts * x2dot)

        return xN.reshape((stateSize, 1))


class AUV(BaseBenchmark):
    def __init__(self, enableNoise):
        super().__init__(0.02, enableNoise)

        self.stateSize = 5
        self.inputSize = 4
        self.paraSize = 4

    @partial(jax.jit, static_argnums=(0,))
    def innerDynamic(self, xT, uT, para):
        xT = xT.reshape((self.stateSize, 1))
        uT = uT.reshape((self.inputSize, ))
        para = para.reshape((self.paraSize, ))
        m = 500.0
        Jz = 300.0
        Xu = 6.106
        Xuu = 5.0
        Yv = 11.203
        Yvv = 10.114
        Nr = 210.0
        Nrr = 3.0
        l1x = -1.01
        l1y = -0.353
        alpha1 = 0.7853981633974483
        l2x = -1.01
        l2y = 0.353
        alpha2 = -0.7853981633974483
        l3x = 1.01
        l3y = -0.353
        alpha3 = -0.7853981633974483
        l4x = 1.01
        l4y = 0.353
        alpha4 = 0.7853981633974483
        # h1=uT[4]#*0+1
        # h2=uT[5]##*0+1
        # h3=uT[6]#*0+1
        # h4=uT[7]#*0+1
        h1 = para[0]
        h2 = para[1]
        h3 = para[2]
        h4 = para[3]
        # B=rho*9.81*(NablaH+uT[0])
        pG = 38
        F1_y = jnp.cos(alpha1) * uT[0] * pG
        F2_y = jnp.cos(alpha2) * uT[1] * pG
        F3_y = jnp.cos(alpha3) * uT[2] * pG
        F4_y = jnp.cos(alpha4) * uT[3] * pG

        F1_x = jnp.sin(alpha1) * uT[0] * pG
        F2_x = jnp.sin(alpha2) * uT[1] * pG
        F3_x = jnp.sin(alpha3) * uT[2] * pG
        F4_x = jnp.sin(alpha4) * uT[3] * pG

        x1dot = 1 / m * (
                    -Xu * xT[0] - Xuu * xT[0] ** 2 + m * xT[1] * xT[2] + h1 * F1_x + F2_x * h2 + F3_x * h3 + F4_x * h4)
        x2dot = 1 / m * (
                    -Yv * xT[1] - Yvv * xT[1] ** 2 - m * xT[0] * xT[2] + h1 * F1_y + h2 * F2_y + h3 * F3_y + h4 * F4_y)
        x3dot = 1 / Jz * (-Nr * xT[2] - Nrr * xT[2] ** 2 + h1 * (-F1_x * l1y + F1_y * l1x) +
                          h2 * (-F2_x * l2y + F2_y * l2x) + h3 * (-F3_x * l3y + F3_y * l3x) + h4 * (
                                      -F4_x * l4y + F4_y * l4x))
        x4dot = xT[2]
        x5dot = xT[3] - 0.2

        Ts = .01
        xN = jnp.zeros((self.stateSize, 1))
        xN = xN.at[0].set(xT[0] + Ts * x1dot)
        xN = xN.at[1].set(xT[1] + Ts * x2dot)
        xN = xN.at[2].set(xT[2] + Ts * x3dot)
        xN = xN.at[3].set(xT[3] + Ts * x4dot)
        xN = xN.at[4].set(xT[4] + Ts * x5dot)

        return xN.reshape((self.stateSize, 1))


class squaredTank(BaseBenchmark):
    def __init__(self, enableNoise):
        super().__init__(0.02, enableNoise)

        global stateSize
        assert stateSize == 2
        global inputSize
        assert inputSize == 1
        pass

    @partial(jax.jit, static_argnums=(0,))
    def innerDynamic(self, xT, uT, p):
        uT = uT.reshape((inputSize))
        xT = xT.reshape((stateSize,))
        xT = jnp.maximum(xT, xT * 0);
        xN = jnp.zeros((stateSize, 1))
        d0 = -jnp.sqrt(xT[0]) * 0.5 + 0.4 * jnp.sign(uT[0]) * (uT[0] ** 2)
        d1 = +jnp.sqrt(xT[0]) * 0.2 - jnp.sqrt(xT[1]) * 0.3
        xN = xN.at[0].set(xT[0] + d0.reshape((1,)))
        xN = xN.at[1].set(xT[1] + d1.reshape((1,)))
        # xN=jnp.maximum(xN,xN*0);
        return xN.reshape((stateSize, 1))

# %%