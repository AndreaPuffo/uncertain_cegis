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
    
    
    x=onp.zeros((stateSize));
    u=onp.zeros((inputSize));
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
def Khotare():
    pass
    import cvxpy as cp
    from halo import HALO
    
    
    max_feval = 8000  # maximum number of function evaluations
    max_iter = 8000  # maximum number of iterations
    beta = 1e-2  # beta controls the usage of the local optimizers during the optimization process
    # With a lower value of beta HALO will use the local search more rarely and viceversa.
    # The parameter beta must be less than or equal to 1e-2 or greater than equal to 1e-4.
    local_optimizer = 'L-BFGS-B' # Choice of local optimizer from scipy python library.
    # The following optimizers are available: 'L-BFGS-B', 'Nelder-Mead', 'TNC' and 'Powell'.
    # For more infomation about the local optimizers please refer the scipy documentation.
    verbose = 0  # this controls the verbosity level, fixed to 0 no output of the optimization progress 
    # will be printed.
    
    
    
    x=onp.zeros((stateSize))+0.5;
    u=onp.zeros((inputSize))+0.5;
    p=onp.ones((paraSize))
    setOfVertices=[computeAB(x,u,p)]
    
    while(True):
        def synthesizeController():
            Qx=onp.eye(stateSize)/1
            R=onp.eye(inputSize)/1   #sqrt in realtà
            
            W=cp.Variable((stateSize,stateSize), symmetric=True)
            X=cp.Variable((inputSize,inputSize), symmetric=True)
            Q=cp.Variable((inputSize,stateSize))
            gamma=cp.Variable((1,1))
            
            A,B=setOfVertices[0]
            constraints = [W >> 1*onp.eye(stateSize),
                            W<<100*onp.eye(stateSize),
                            # cp.vstack([ cp.hstack([onp.eye(1),      onp.ones((stateSize,1)).T]),
                            #             cp.hstack([onp.ones((stateSize,1)),W])                               
                            #   ])>>0
                           # cp.vstack([ cp.hstack([-X,      (R@Q)]),
                           #             cp.hstack([(R@Q).T, -W])                               
                           #   ])<<0
                           ]
            L=onp.diag(1/Bounds.ub[:stateSize])
            for k in range(stateSize):
                Lr=L[k:k+1,:]
                print(Lr.shape)
                constraints+=[                
                    cp.vstack([cp.hstack([onp.eye(1),   Lr@W]),
                                cp.hstack([ W@Lr.T,     W])])>>0
                    ]
                
            for AB in setOfVertices:
                A,B=AB
                constraints += [                    
                    # cp.vstack([cp.hstack([-W+onp.eye(stateSize),      A@W+B@Q]),
                    #             cp.hstack([(A@W+B@Q).T,-W])                            
                    #   ])<<-1e-2
                    cp.vstack([cp.hstack([W,      (A@W+B@Q).T,W@Qx,Q.T@R]),
                                cp.hstack([(A@W+B@Q),W,W*0,Q.T*0]),
                                cp.hstack([(W@Qx).T,W*0,gamma*onp.eyMastie(stateSize),Q.T*0]),
                                cp.hstack([(Q.T@R).T,Q*0,Q*0,gamma*onp.eye(inputSize)])                           
                      ])>>1e-2
                    
                ]
            prob = cp.Problem(cp.Minimize(gamma/1000), constraints)
            prob.solve(solver='MOSEK',verbose=True)
            
            print("The optimal value is", prob.value)
            # print("A solution X is")
            P=onp.linalg.inv(W.value);
            K=(Q.value@onp.linalg.inv(W.value))
            
            for AB in setOfVertices:
                A,B=AB
                print(onp.linalg.eigvals(A+B@K))
                assert(onp.all(onp.abs(onp.linalg.eigvals(A+B@K))<1))
            return P,K
        
        P,K=synthesizeController()
        
        
        
        
        def costEigPK(x,P,K):
            x=onp.reshape(x, (1,stateSize+inputSize+1))
            # A,B=approximateOnPointJAC(x,nbrs)
            xState=x[0:1,0:stateSize]
            u=x[0:1,stateSize:stateSize+inputSize]
            p=x[0:1,stateSize+inputSize:]
            # xTT=system.forward(u,xState,p)
            # overallDataset+=[(xState.T,u[0],xTT,computeAB(xState.T,u))]
            v=onp.Inf;
            for k in range(paraSize):
                para_v=onp.ones((paraSize,1));
                para_v[k]=p[0:1];
                A,B=computeAB(xState.T,u,para_v)
                eig=scipy.sparse.linalg.eigsh(            
                    onp.vstack([onp.hstack([P,(A+B@K).T@P]),
                                onp.hstack([P@(A+B@K),P])]),1,which='SA',return_eigenvectors=False)
                # print(eig)
                v=min(min(onp.real(eig)),v)
            return v
        
        costEig=lambda x: costEigPK(x,P,K)
        
        res=scipy.optimize.direct(costEig,bounds=Bounds,locally_biased=True) 
        # res=scipy.optimize.minimize(costEig,res.x,bounds=Bounds) 
        # halo = HALO(costEig, [[Bounds.lb[i],Bounds.ub[i]] for i in range(0,len(Bounds.lb))], max_feval, max_iter, beta, local_optimizer, verbose)
        # result=halo.minimize();
        result={'best_f':res.fun,'best_x':res.x}
        print(result['best_f'])
        if result['best_f']>=0:
            break
        else:
            x=result['best_x']
            x=onp.reshape(x, (1,stateSize+inputSize+1))
            xState=x[0:1,0:stateSize]
            u=x[0:1,stateSize:stateSize+inputSize]
            p=x[0:1,stateSize+inputSize:]
            for k in range(paraSize):
                para_v=onp.ones((paraSize));
                para_v[k]=p[0:1];
                setOfVertices+=[computeAB(xState,u,para_v)]
    
    return K,P
    


