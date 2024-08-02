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

import jax.experimental
# config.update('jax_disable_jit', True)
regWeight=0.01
threshold=.51
n_neighbors=5
#%%
class BaseBenchmark:
    def __init__(self,noiseCov=None,enableNoise=True):
        self.enableNoise=enableNoise
        if noiseCov is None:
            self.noiseCov=0.1;
        else:
            self.noiseCov=noiseCov;
            print('overwritten noise cov:',noiseCov)
        pass
    def forward(self,uT,x0=None,norm=False):
        global stateSize
        global inputSize        
        x0=onp.reshape(x0,(stateSize,1))
        uT=onp.reshape(uT,(inputSize,1))
        xN=self.innerDynamic(x0,uT)
        xN=onp.reshape(xN,(stateSize,1))
            
        return xN
    def innerDynamic(x0,uT):
        pass
    
        
#%%
        
class quasiLPV(BaseBenchmark):
    def __init__(self,enableNoise):       
        super().__init__(0.02,enableNoise)
        
        global stateSize
        assert stateSize==3
        global inputSize
        assert inputSize==2
        self.norm=None
        self.x0=onp.array([0,0,0]).reshape((3,1))
        self.A=onp.array([[-0.29181893, -0.05231634,  0.        ],
                           [ 0.11910899, -1.48769104, -0.43707303],
                           [-0.5       ,  0.56982177,  0.4038157 ]])/1.5
        self.B=onp.array([[-1.31637896, 1.69459169],
                           [0.37108047, 0.3295812 ],
                           [1.30842492, -1.04712173]])
        
        pass
    
    @partial(jax.jit, static_argnums=(0,))
    def innerDynamic(self,xT, uT):
        uT=uT.reshape((inputSize,1))
        xT=xT.reshape((stateSize,1))
        xN=jnp.zeros((stateSize,1))        
        xN=self.A@xT+jnp.tanh(xT)+self.B@uT+jnp.maximum(self.B@uT,0)
        # xN=jnp.maximum(xN,xN*0);
        
        return xN.reshape((stateSize,1))
    

 
 
    
        
class TxT(BaseBenchmark):
    def __init__(self,enableNoise):       
        super().__init__(0.02,enableNoise)
        
        global stateSize
        assert stateSize==1
        global inputSize
        assert inputSize==1
        self.norm=None
        self.x0=onp.array([0]).reshape((1,1))
        self.A=onp.array([[.69181893]])
        self.B=onp.array([[-1.31637896]])
        
        pass
    
    @partial(jax.jit, static_argnums=(0,))
    def innerDynamic(self,xT, uT):
        uT=uT.reshape((inputSize,1))
        xN=jnp.zeros((stateSize,1))        
        xN=self.A@xT+jnp.tanh(xT)+self.B@uT+jnp.maximum(self.B@uT,0)-1.55*jnp.maximum(-self.B@(uT+0.5),0)
        # xN=jnp.maximum(xN,xN*0);
        
        return xN.reshape((stateSize,1))
    

 

        
class glider(BaseBenchmark):
    def __init__(self,enableNoise):       
        super().__init__(0.02,enableNoise)
        
        global stateSize
        assert stateSize==2
        global inputSize
        assert inputSize==2

        pass
    
    @partial(jax.jit, static_argnums=(0,))
    def innerDynamic(self,xT, uT):
        ms=44.9
        mp=11
        m1=64.84
        m3=99.43
        KD0=0.04
        KD=9.44
        KL0=2.16
        KL=4.88
        NablaH=0.054
        rho=1027.5
        KFD=10
        KUDelta=1
        alpha=-3.5/360*onp.pi*2
        theta=23.7/360.0*onp.pi*2
        uT=uT.reshape((inputSize))
        xT=xT.reshape((stateSize))
        Vq=jnp.sum(xT**2)
        D=(KD0+KD*alpha**2)*Vq
        L=(KL0+KL*alpha)*Vq
        # B=rho*9.81*(NablaH+uT[0])
        B=uT[0]*1
        # Fdelta1=KFD*KUDelta*Vq*uT[1]
        # Fdelta2=KFD*KUDelta*Vq*uT[2]
        Fdelta1=uT[1]
        Fdelta2=uT[2]
        v1dot=1/m1*(-D*jnp.cos(alpha)+L*jnp.sin(alpha)+jnp.sin(theta)*(B-9.81*ms-9.81*mp))
        v2dot=1/m3*(-D*jnp.sin(alpha)-L*jnp.cos(alpha)+jnp.cos(theta)*(-B+9.81*ms+9.81*mp)+Fdelta1+Fdelta2)
        Ts=0.125
        xN=jnp.zeros((stateSize,1))        
        xN=xN.at[0].set(xT[0]+Ts*v1dot)
        xN=xN.at[1].set(xT[1]+Ts*v2dot)
        return xN.reshape((stateSize,1))

        
class squaredTank(BaseBenchmark):
    def __init__(self,enableNoise):       
        super().__init__(0.02,enableNoise)
        
        global stateSize
        assert stateSize==2
        global inputSize
        assert inputSize==1        
        pass
    
    @partial(jax.jit, static_argnums=(0,))
    def innerDynamic(self,xT, uT):
        uT=uT.reshape((inputSize))
        xT=xT.reshape((stateSize,))
        xT=jnp.maximum(xT,xT*0);
        xN=jnp.zeros((stateSize,1))        
        d0=-jnp.sqrt(xT[0])*0.5+0.4*jnp.sign(uT[0])*(uT[0]**2)
        d1=+jnp.sqrt(xT[0])*0.2-jnp.sqrt(xT[1])*0.3
        xN=xN.at[0].set(xT[0]+d0.reshape((1,)))
        xN=xN.at[1].set(xT[1]+d1.reshape((1,)))
        # xN=jnp.maximum(xN,xN*0);
        return xN.reshape((stateSize,1))

#%%        
    

import sys
benchamark_id= 4
       
switch_dict = {    
    1: lambda: (squaredTank,2,1,scipy.optimize.Bounds(onp.ones((3,))*0.01,onp.ones((3,))*0+5)),
    2: lambda: (glider,2,2,scipy.optimize.Bounds(onp.array([-1,-1,-1,-onp.pi/8]),onp.array([1,1,1,onp.pi/8]))),
    3: lambda: (quasiLPV,3,2,scipy.optimize.Bounds(onp.ones((5,))*-3,onp.ones((5,))*3)),
    4: lambda: (TxT,1,1,scipy.optimize.Bounds(-onp.ones((2,))*7,onp.ones((2,))*7))
}
     
systemClass,stateSize,inputSize,Bounds=switch_dict.get(benchamark_id)()



system=systemClass(enableNoise=False)
onp.random.seed(0)
jacA=jax.jit(jax.jacobian(system.innerDynamic))
jacB=jax.jit(jax.jacobian(system.innerDynamic,argnums=1))

computeAB=lambda x,u: [jacA(x,u).reshape((stateSize,stateSize)),jacB(x,u).reshape((stateSize,inputSize))]


#%%

import cvxpy as cp
from halo import HALO


max_feval = 8000  # maximum number of function evaluations
max_iter = 8000  # maximum number of iterations
beta = 3e-3  # beta controls the usage of the local optimizers during the optimization process
# With a lower value of beta HALO will use the local search more rarely and viceversa.
# The parameter beta must be less than or equal to 1e-2 or greater than equal to 1e-4.
local_optimizer = 'Nelder-Mead' # Choice of local optimizer from scipy python library.
# The following optimizers are available: 'L-BFGS-B', 'Nelder-Mead', 'TNC' and 'Powell'.
# For more infomation about the local optimizers please refer the scipy documentation.
verbose = 0  # this controls the verbosity level, fixed to 0 no output of the optimization progress 
# will be printed.



x=onp.zeros((stateSize))+0.1;
u=onp.zeros((inputSize))+0.1
setOfVertices=[computeAB(x,u)]

while(True):
    def synthesizeController():
        Qx=onp.eye(stateSize)
        R=onp.eye(inputSize)   #sqrt in realtà
        
        W=cp.Variable((stateSize,stateSize), symmetric=True)
        X=cp.Variable((inputSize,inputSize), symmetric=True)
        Q=cp.Variable((inputSize,stateSize))
        # gamma=cp.Variable((1,1))
        
        
        constraints = [W >> onp.eye(stateSize)]
        for AB in setOfVertices:
            A,B=AB
            constraints += [
                    cp.vstack([ cp.hstack([X,      (R@Q)]),
                                cp.hstack([(R@Q).T, W])                               
                      ])>>1e-4,
                cp.vstack([cp.hstack([W-onp.eye(stateSize),      A@W+B@Q]),
                            cp.hstack([(A@W+B@Q).T,W])                            
                  ])>>0,
            ]
        prob = cp.Problem(cp.Minimize(cp.trace(X)+cp.trace(Qx@W)), constraints)
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
        x=onp.reshape(x, (1,stateSize+inputSize))
        # A,B=approximateOnPointJAC(x,nbrs)
        xState=x[0:1,0:stateSize]
        u=x[0:1,stateSize:]
        xTT=system.forward(u,xState)
        # overallDataset+=[(xState.T,u[0],xTT,computeAB(xState.T,u))]
        A,B=computeAB(xState.T,u)
        eig=scipy.sparse.linalg.eigsh(            
            onp.vstack([onp.hstack([P,(A+B@K).T@P]),
                        onp.hstack([P@(A+B@K),P])]),1,sigma=0,return_eigenvectors=False)
        # print(eig)
        return min(onp.real(eig))
    
    costEig=lambda x: costEigPK(x,P,K)
    local_optimizer = 'L-BFGS-B' 
    # res=scipy.optimize.direct(costEig,bounds=Bounds,locally_biased=True,maxiter=80000,maxfun=80000    ) 
    # res=scipy.optimize.minimize(costEig,onp.zeros((stateSize+inputSize,)),bounds=Bounds) 
    halo = HALO(costEig, [[Bounds.lb[i],Bounds.ub[i]] for i in range(0,len(Bounds.lb))], max_feval, max_iter, beta, local_optimizer, verbose)
    result=halo.minimize();
    print(result['best_f'])
    if result['best_f']>=0:
        break
    else:
        x=result['best_x']
        x=onp.reshape(x, (1,stateSize+inputSize))
        xState=x[0:1,0:stateSize]
        u=x[0:1,stateSize:]
        setOfVertices+=[computeAB(xState,u)]



