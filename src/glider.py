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
from cegis import CegisBemporad
    

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
def simulateController(K,label,ax1,ax2,x0=None):
    story=[]
    if x0 is None:
        xState=onp.zeros((1,stateSize))
        xState[0,0]=0
        xState[0,1]=0
    else:
        xState=onp.reshape(x0,(1,stateSize))
    p=onp.ones((1,paraSize))
    for k in range(0,35000):
        ref=onp.array(xState*0)
        ref[0,0]=onp.cos(k/1000)/5+0.5
        ref[0,1]=onp.cos((600+k)/1000)/5
        # err=
        # xState[0,-1]+=-.1
        uF=onp.clip((K@(xState-ref).T).ravel(),Bounds.lb[stateSize:stateSize+inputSize],Bounds.ub[stateSize:stateSize+inputSize])
        # uF=onp.array(K@(xState-ref).T).ravel()
        
        xN=system.innerDynamic(xState,uF,p)
        p=onp.ones((1,paraSize))
        if k>=3000 and k<=4000:
            p[0,2]=.1
        if k>=5000:
            p[0,1]=.1
        else:
            pass
        story+=[(xState,uF)]
        xState=onp.array(onp.reshape(xN,(1,stateSize)))
        
        # print(xState)
    
        
    import matplotlib.pyplot as plt
    
    ax1.title.set_text('input '+str(label))
    ax1.plot(onp.array([x[1] for x in story]).reshape((-1,inputSize)))
    
    ax2.title.set_text('state '+str(label))
    ax2.plot(onp.array([x[0] for x in story]).reshape((-1,stateSize)))
    
# plt.plot(onp.array([x[0].T@P@x[0] for x in story]).reshape((-1,stateSize)))
# plt.figure()
fig, (ax1, ax2)=plt.subplots(1, 2, sharey=False)
simulateController(Ksat,'K sat',ax1,ax2)
# plt.figure()
if benchamark_id==6:
    fig, (ax1, ax2)=plt.subplots(1, 2, sharey=False)
    simulateController(Kinf,'$H_{inf}$',ax1,ax2)

    #%%
    PKinf=computeEllipsoid(Kinf)
    import numpy as np
    #%%
    plt.figure()
    
    # Define the symmetric matrix Q
    def printEllipse(Q,colorString,label):
        
        # Q=np.linalg.inv(Psat)
        # Create a grid of points
        x = np.linspace(-2, 2, 100)
        y = np.linspace(-2, 2, 100)
        X, Y = np.meshgrid(x, y)
        Z = np.zeros_like(X)
        
        # Compute the quadratic form values on the grid
        for i in range(X.shape[0]):
            for j in range(X.shape[1]):
                vec = np.array([X[i, j], Y[i, j]]).reshape((2,1))
                Z[i, j] = vec.T@Q@vec
                
        contour=plt.contour(X, Y, Z, levels=[1], colors=colorString)
        plt.xlabel('x1')
        plt.ylabel('x2')
        plt.title('Contour plot of $x^T Q x = 1$')
        # plt.gca().set_aspect('equal', adjustable='box')
        plt.clabel(contour,contour.levels,fmt= lambda x: str(label))
        # plt.colorbar(contour)
    printEllipse(Psat,'b','Psat')
    printEllipse(PKinf[0],'r','$H_{inf}$')
    plt.show()


#%%
def H2():
    # pass
    import cvxpy as cp
    
    
    x=onp.zeros((stateSize))
    u=onp.zeros((inputSize))
    p=onp.ones((paraSize))
    setOfVertices=[computeAB(x,u,p)]
    

    def synthesizeController():
        Qx=onp.eye(stateSize)*1e1
        R=onp.eye(inputSize)/1e1   #sqrt in realtà
        
        X=cp.Variable((stateSize,stateSize), symmetric=True)
        Y=cp.Variable((inputSize,inputSize), symmetric=True)
        W=cp.Variable((inputSize,stateSize))
        gamma=cp.Variable((1,1))
        
        A,B=setOfVertices[0]
        constraints = [X >> onp.eye(stateSize),  #Y>>0, gamma<=0  ,              
                       cp.vstack([ cp.hstack([-Y,      (R@W)]),
                                   cp.hstack([(R@W).T, -X])                               
                         ])<<0,                  
            cp.vstack([cp.hstack([-X+onp.eye(stateSize),      A@X+B@W]),
                        cp.hstack([(A@X+B@W).T,-X])                            
              ])<<0
            
        ]
        prob = cp.Problem(cp.Minimize(cp.trace(Qx@X@Qx.T)+cp.trace(Y)), constraints)
        prob.solve(solver='MOSEK',verbose=True)
        
        print("The optimal value is", prob.value)
        # print("A solution X is")
        P=onp.linalg.inv(X.value);
        K=(W.value@P)
        
        for AB in setOfVertices:
            A,B=AB
            print(onp.linalg.eigvals(A+B@K))
            assert(onp.all(onp.abs(onp.linalg.eigvals(A+B@K))<1))
        return P,K
        
    P,K=synthesizeController()
    
    return P,K
if benchamark_id==5:
    PH2,KH2=H2()
    fig, (ax1, ax2)=plt.subplots(1, 2, sharey=False)
    simulateController(KH2,'$H_{2} from 2 $',ax1,ax2,2*onp.ones((1,stateSize)))
    fig, (ax1, ax2)=plt.subplots(1, 2, sharey=False)
    simulateController(KH2,'$H_{2}$ from 0',ax1,ax2)
    fig, (ax1, ax2)=plt.subplots(1, 2, sharey=False)
    simulateController(Ksat,'$K sat from 2$ from 0',ax1,ax2,2*onp.ones((1,stateSize)))
    # PKH2=computeEllipsoid(KH2)

#%%

    
plt.show()


