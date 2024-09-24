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
#%%
from benchmarks import glider, AUV, twoStateAUV
from cegis import CegisBemporad, CegisEllipsoid
from ctrl_synthesis import controllerH2
from plotting import simulateController, plotEllipse
    

import sys
benchamark_id= 5
b=2
switch_dict = {    
    # 1: lambda: (squaredTank,2,1,0,scipy.optimize.Bounds(onp.ones((3,))*0.01,onp.ones((3,))*0+5)),
    2: lambda: (glider,2,2,1,scipy.optimize.Bounds(onp.array([-1,-1,-1,-onp.pi/8,-5]),onp.array([1,1,1,onp.pi/8,5])),onp.eye(4)),
    # 3: lambda: (quasiLPV,3,2,0,scipy.optimize.Bounds(onp.ones((5,))*-3,onp.ones((5,))*3)),
    # 4: lambda: (TxT,1,1,0,scipy.optimize.Bounds(-onp.ones((2,))*7,onp.ones((2,))*7)),
    5: lambda: (AUV,5,4,4,scipy.optimize.Bounds(
                            onp.array([-b,-b,-b,-b,-b,-1,-1,-1,-1,0.0]),
                            onp.array([b,b,b,b,b,1,1,1,1,1])),onp.delete(onp.diag([1,1,1,0,0,1,1,1,1,1]),[3,4],axis=0).T),
    6: lambda: (twoStateAUV,2,3,3,scipy.optimize.Bounds(
                            onp.array([-b,-b,-38,-38,-38,0.0]),
                            onp.array([b,b,38,38,38,1])),onp.eye(5))
}
     
systemClass,stateSize,inputSize,paraSize,Bounds,mask=switch_dict.get(benchamark_id)()

system=systemClass(enableNoise=False)
onp.random.seed(0)
jacA=jax.jit(jax.jacobian(system.innerDynamic))
jacB=jax.jit(jax.jacobian(system.innerDynamic,argnums=1))

# compute jacobians on the fly, returning A and B matrices
computeAB=lambda x,u,p: [jacA(x,u,p).reshape((stateSize,stateSize)),jacB(x,u,p).reshape((stateSize,inputSize))]


#%%
tau=1-0.001
# compute the P and K matrix via CEGIS
Psat,Ksat = CegisBemporad(benchmark=system, computeAB=computeAB, matrixbounds=Bounds, mask=mask, tau=tau)

#%%
P=Psat*1
K=Ksat*1
#%%
K_Hinf2= 1.0e+04*onp.array([[0.2400,    0.2865],
                         [0.2553,   -0.3037],
                         [-0.0013,   -2.2510]])
K_Hinf1= onp.array([[232.1081,  277.2772],
                       [183.4074, -219.4158],
                       [-0.0298, -776.4082]])

Kinf=-K_Hinf1

#%%
# plt.plot(onp.array([x[0].T@P@x[0] for x in story]).reshape((-1,stateSize)))
# plt.figure()
fig, (ax1, ax2)=plt.subplots(1, 2, sharey=False)
simulateController(system, Ksat, horizon=35000, paraSize=paraSize, variableBounds=Bounds,
                   label='K sat', axes=[ax1,ax2])
# plt.figure()
if benchamark_id==6:
    fig, (ax1, ax2)=plt.subplots(1, 2, sharey=False)
    simulateController(system, Kinf, horizon=35000, paraSize=paraSize, variableBounds=Bounds,
                       label='$H_{inf}$', axes=[ax1,ax2])

    PKinf = CegisEllipsoid(Kinf, system, computeAB, Bounds, mask, tau)

    plt.figure()
    plotEllipse(Psat,'b','Psat')
    plotEllipse(PKinf[0],'r','$H_{inf}$')
    plt.show()


#%%

if benchamark_id==5:
    PH2, KH2 = controllerH2(stateSize, inputSize, paraSize, computeAB)
    fig, (ax1, ax2)=plt.subplots(1, 2, sharey=False)
    simulateController(system, KH2, horizon=35000, paraSize=paraSize, variableBounds=Bounds,
                       label=r'$H_{2}$ from 2', axes=[ax1,ax2], x0=2*onp.ones((1,stateSize)))

    fig, (ax1, ax2)=plt.subplots(1, 2, sharey=False)
    simulateController(system, KH2, horizon=35000, paraSize=paraSize, variableBounds=Bounds,
                       label=r'$H_{2}$ from 0', axes=[ax1,ax2])

    fig, (ax1, ax2)=plt.subplots(1, 2, sharey=False)
    simulateController(system, Ksat, horizon=35000, paraSize=paraSize, variableBounds=Bounds,
                       label=r'Bemporad $K_{sat}$ from 2', axes=[ax1,ax2], x0=2*onp.ones((1,stateSize)))
    # PKH2=computeEllipsoid(KH2)

#%%

    
plt.show()


