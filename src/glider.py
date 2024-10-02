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
class BaseBenchmark:
    def __init__(self,noiseCov=None,enableNoise=True):
        self.enableNoise=enableNoise
        if noiseCov is None:
            self.noiseCov=0.1;
        else:
            self.noiseCov=noiseCov;
            print('overwritten noise cov:',noiseCov)
        pass

    def innerDynamic(x0,uT,p):
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
    def innerDynamic(self,xT, uT,p):
        uT=uT.reshape((inputSize,1))
        xT=xT.reshape((stateSize,1))
        xN=jnp.zeros((stateSize,1))        
        xN=self.A@xT+jnp.tanh(xT)+self.B@uT+jnp.maximum(self.B@uT,0)        
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
    def innerDynamic(self,xT, uT,p):
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
        global paraSize
        assert paraSize==1

        pass
    
    @partial(jax.jit, static_argnums=(0,))
    def innerDynamic(self,xT, uT,p):
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
        alpha=(-3.5+ p[0])/360*onp.pi*2
        theta=(23.7+p[0])/360.0*onp.pi*2
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




        
class twoStateAUV(BaseBenchmark):
    def __init__(self,enableNoise):       
        super().__init__(0.02,enableNoise)
        
        global stateSize
        assert stateSize==2
        global inputSize
        assert inputSize==3
        global paraSize
        assert paraSize==3

        pass
    
    @partial(jax.jit, static_argnums=(0,))
    def innerDynamic(self,xT, uT,para):
        xT=xT.reshape((stateSize,1))
        uT=uT.reshape((inputSize))
        para=para.reshape((paraSize))
        m = 500.0
        Jz= 300.0
        Xu = 6.106
        Xuu = 5.0
        Nr=210.0
        Nrr=3.0
        l1x=1.01
        l1y=-0.353
        alpha1=1.9198621771937625
        l2x=-1.01
        l2y= 0.353
        alpha2=1.2217304763960306
        l3x =0.75 
        l3y=0.0
        alpha3 =3.141592653589793
        h1=para[0]
        h2=para[1]
        h3=para[2]
        
        
        pG=1#40 # actuator range here so thatr sat bound is -1/+1
        F1_x=jnp.sin(alpha1)*uT[0]*pG
        F2_x=jnp.sin(alpha2)*uT[1]*pG
        F3_x=jnp.sin(alpha3)*uT[2]*pG
        
        
        F1_y=jnp.cos(alpha1)*uT[0]*pG
        F2_y=jnp.cos(alpha2)*uT[1]*pG
        F3_y=jnp.cos(alpha3)*uT[2]*pG
        
        
        x1dot=1/m*(-Xu*xT[0]-Xuu*xT[0]**2+h1*F1_x+F2_x*h2+F3_x*h3)
        
        x2dot=1/Jz*(-Nr*xT[1]-Nrr*xT[1]**2+h1*(-F1_x*l1y+F1_y*l1x)+
                     h2*(-F2_x*l2y+F2_y*l2x)+h3*(-F3_x*l3y+F3_y*l3x))
        
        
        Ts=.01
        xN=jnp.zeros((stateSize,1))        
        xN=xN.at[0].set(xT[0]+Ts*x1dot)
        xN=xN.at[1].set(xT[1]+Ts*x2dot)
        
        
        return xN.reshape((stateSize,1))





        
class AUV(BaseBenchmark):
    def __init__(self,enableNoise):       
        super().__init__(0.02,enableNoise)
        
        global stateSize
        assert stateSize==5
        global inputSize
        assert inputSize==4
        global paraSize
        assert paraSize==4

        pass
    
    @partial(jax.jit, static_argnums=(0,))
    def innerDynamic(self,xT, uT,para):
        xT=xT.reshape((stateSize,1))
        uT=uT.reshape((inputSize))
        para=para.reshape((paraSize))
        m= 500.0
        Jz= 300.0
        Xu= 6.106 
        Xuu= 5.0
        Yv= 11.203 
        Yvv= 10.114
        Nr= 210.0
        Nrr= 3.0 
        l1x= -1.01 
        l1y= -0.353
        alpha1= 0.7853981633974483
        l2x= -1.01 
        l2y= 0.353 
        alpha2= -0.7853981633974483
        l3x= 1.01
        l3y= -0.353 
        alpha3= -0.7853981633974483
        l4x= 1.01 
        l4y= 0.353 
        alpha4= 0.7853981633974483
        # h1=uT[4]#*0+1
        # h2=uT[5]##*0+1
        # h3=uT[6]#*0+1
        # h4=uT[7]#*0+1
        h1=para[0]
        h2=para[1]
        h3=para[2]
        h4=para[3]
        # B=rho*9.81*(NablaH+uT[0])
        pG=38
        F1_y=jnp.cos(alpha1)*uT[0]*pG
        F2_y=jnp.cos(alpha2)*uT[1]*pG
        F3_y=jnp.cos(alpha3)*uT[2]*pG
        F4_y=jnp.cos(alpha4)*uT[3]*pG
        
        F1_x=jnp.sin(alpha1)*uT[0]*pG
        F2_x=jnp.sin(alpha2)*uT[1]*pG
        F3_x=jnp.sin(alpha3)*uT[2]*pG
        F4_x=jnp.sin(alpha4)*uT[3]*pG
        
        x1dot=1/m*(-Xu*xT[0]-Xuu*xT[0]**2+m*xT[1]*xT[2]+h1*F1_x+F2_x*h2+F3_x*h3+F4_x*h4)
        x2dot=1/m*(-Yv*xT[1]-Yvv*xT[1]**2-m*xT[0]*xT[2]+h1*F1_y+h2*F2_y+h3*F3_y+h4*F4_y)
        x3dot=1/Jz*(-Nr*xT[2]-Nrr*xT[2]**2+h1*(-F1_x*l1y+F1_y*l1x)+
                     h2*(-F2_x*l2y+F2_y*l2x)+h3*(-F3_x*l3y+F3_y*l3x)+h4*(-F4_x*l4y+F4_y*l4x))
        x4dot=xT[2]
        x5dot=xT[3]-0.2
        
        Ts=.01
        xN=jnp.zeros((stateSize,1))        
        xN=xN.at[0].set(xT[0]+Ts*x1dot)
        xN=xN.at[1].set(xT[1]+Ts*x2dot)
        xN=xN.at[2].set(xT[2]+Ts*x3dot)
        xN=xN.at[3].set(xT[3]+Ts*x4dot)
        xN=xN.at[4].set(xT[4]+Ts*x5dot)
        
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
    def innerDynamic(self,xT, uT,p):
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
                            onp.array([b,b,38,38,38,1])),onp.eye(6))
}
     
systemClass,stateSize,inputSize,paraSize,Bounds,mask=switch_dict.get(benchamark_id)()



system=systemClass(enableNoise=False)
onp.random.seed(0)
jacA=jax.jit(jax.jacobian(system.innerDynamic))
jacB=jax.jit(jax.jacobian(system.innerDynamic,argnums=1))

computeAB=lambda x,u,p: [jacA(x,u,p).reshape((stateSize,stateSize)),jacB(x,u,p).reshape((stateSize,inputSize))]


#%%
tau=1-0.001
def Bemporad():
    import cvxpy as cp
   
    x=onp.zeros((stateSize))+1;
    u=onp.zeros((inputSize))+1
    p=onp.ones((paraSize))
    setOfVertices=[computeAB(x,u,p)]
    print(setOfVertices)
    
    while(True):
        def synthesizeController():
           
            
            Q=cp.Variable((stateSize,stateSize), symmetric=True)
            Dw=onp.eye(stateSize)*0
            Y=cp.Variable((inputSize,stateSize))
            Z=cp.Variable((inputSize,stateSize))        
            
            
            
            constraints = [  ]
            for k in range(inputSize):
                Zr=Z[k:k+1,:]
                # print(Zr.shape)
                constraints+=[                
                    cp.vstack([cp.hstack([onp.eye(1)*(Bounds.ub[stateSize+k]**2),   Zr]),
                                cp.hstack([ Zr.T,     Q])])>>0
                    ]
                
            L=onp.diag(1/Bounds.ub[:stateSize])
            for k in range(stateSize):
                Lr=L[k:k+1,:]
                # print(Lr.shape)
                constraints+=[                
                    cp.vstack([cp.hstack([onp.eye(1),   Lr@Q]),
                                cp.hstack([ Q@Lr.T,     Q])])>>0
                    ]
                
            
            
            combinations=int(2**inputSize)
            Es=[]
            
            for j in range(combinations):
                Es += [onp.diag([int(b) for b in onp.binary_repr(j, inputSize)])]
            for AB in setOfVertices:
                A,B=AB
                for Ej in Es:           
                    ACL=(A@Q+B@Ej@Y+B@(onp.eye(inputSize)-Ej)@Z)
                    constraints += [                    
                        cp.vstack([cp.hstack([(tau)*Q,      Q*0,ACL.T]),
                                    cp.hstack([Q*0,      onp.eye(stateSize)*(1-tau),Dw]),
                                    cp.hstack([ACL, Dw.T, Q]),
                          ])>>1e-4
                        
                    ]
            prob = cp.Problem(cp.Minimize(-cp.trace(Q)), constraints)
            prob.solve(solver='MOSEK',verbose=False)
            
            print("The optimal value is", prob.value)
            # print("A solution X is")
            P=onp.linalg.inv(Q.value);
            K=(Y.value@P)
            H=Z.value@P 
            for AB in setOfVertices:
                A,B=AB
                print(onp.linalg.eigvals(A+B@K))
                assert(onp.all(onp.abs(onp.linalg.eigvals(A+B@K))<1))
            return P,K,H
        
        P,K,H=synthesizeController()
        
        
        
        
        def costEigPK(x,Q,K,H):
            global mask
            x=onp.reshape(x, (mask.shape[1],1))
            # A,B=approximateOnPointJAC(x,nbrs)
            x=mask@x
            x=x.reshape((1,stateSize+inputSize+1))
            xState=x[0:1,0:stateSize]
            u=x[0:1,stateSize:stateSize+inputSize]
            p=x[0:1,stateSize+inputSize:]
            # xTT=system.forward(u,xState,p)
            # overallDataset+=[(xState.T,u[0],xTT,computeAB(xState.T,u))]
            v=onp.Inf;
            
            Y=K@Q
            Z=H@Q
            Es=[]         
            
            Dw=onp.eye( stateSize)*0
            combinations=int(2**inputSize)
            for j in range(combinations):
                Es += [onp.diag([int(b) for b in onp.binary_repr(j, inputSize)])]
            
            
            for k in range(paraSize):
                para_v=onp.ones((paraSize,1));
                para_v[k]=p[0:1];
                A,B=computeAB(xState.T,u,para_v)            
                for Ej in Es:           
                    ACL=(A@Q+B@Ej@Y+B@(onp.eye(inputSize)-Ej)@Z)
                    cond =onp.vstack([onp.hstack([(tau)*Q,      Q*0,ACL.T]),
                                    onp.hstack([Q*0,      onp.eye(stateSize)*(1-tau),Dw]),
                                    onp.hstack([ACL, Dw.T, Q]),])
                    # print(constraints[-1].shape)
                    # cond=scipy.linalg.block_diag(constraints)
                    eig=scipy.sparse.linalg.eigsh(cond,1,which='SA',return_eigenvectors=False)
                    # print(eig)
                    v=min(min(onp.real(eig)),v)
                    
            return v
        Q=onp.linalg.inv(P)
        costEig=lambda x: costEigPK(x,Q,K,H)
        b=scipy.optimize.Bounds(mask.T@Bounds.lb,mask.T@Bounds.ub)
        # res=scipy.optimize.direct(costEig,bounds=b,locally_biased=True) 
        res=scipy.optimize.shgo(costEig,bounds=b,options={"f_tol":1e-6}) 
        # res=scipy.optimize.minimize(costEig,res.x,bounds=Bounds) 
        # halo = HALO(costEig, [[Bounds.lb[i],Bounds.ub[i]] for i in range(0,len(Bounds.lb))], max_feval, max_iter, beta, local_optimizer, verbose)
        # result=halo.minimize();
        result={'best_f':res.fun,'best_x':res.x}
        # break
        print("veryfier says: ",result['best_f'])
        if result['best_f']>=-1e-9:
            break
        else:
            x=result['best_x']
            x=mask@x
            x=onp.reshape(x, (1,stateSize+inputSize+1))
            xState=x[0:1,0:stateSize]
            u=x[0:1,stateSize:stateSize+inputSize]
            p=x[0:1,stateSize+inputSize:]
            for k in range(paraSize):
                para_v=onp.ones((paraSize));
                para_v[k]=p[0:1];
                setOfVertices+=[computeAB(xState,u,para_v)]
    return P,K
    
    

#%%


def computeEllipsoid(Kext):
    import cvxpy as cp
    
    x=onp.zeros((stateSize))+1;
    u=onp.zeros((inputSize))+1
    p=onp.ones((paraSize))
    setOfVertices=[computeAB(x,u,p)]
    print(setOfVertices)
    
    while(True):
        def synthesizeController():
            nonlocal Kext
            global stateSize
            global inputSize
            Q=cp.Variable((stateSize,stateSize), symmetric=True)
            Dw=onp.eye(stateSize)*0
            Y=cp.Variable((inputSize,stateSize))
            Z=cp.Variable((inputSize,stateSize))        
            
            
            # print((Q@Kext).shape)
            constraints = [Kext@Q==Y  ]
            for k in range(inputSize):
                Zr=Z[k:k+1,:]
                print(Zr.shape)
                constraints+=[                
                    cp.vstack([cp.hstack([onp.eye(1)*(Bounds.ub[stateSize+k]**2),   Zr]),
                                cp.hstack([ Zr.T,     Q])])>>0
                    ]
                
            L=onp.diag(1/Bounds.ub[:stateSize])
            for k in range(stateSize):
                Lr=L[k:k+1,:]
                print(Lr.shape)
                constraints+=[                
                    cp.vstack([cp.hstack([onp.eye(1),   Lr@Q]),
                                cp.hstack([ Q@Lr.T,     Q])])>>1e-4
                    ]
                
            
            
            combinations=int(2**inputSize)
            Es=[]
            
            for j in range(combinations):
                Es += [onp.diag([int(b) for b in onp.binary_repr(j, inputSize)])]
            for AB in setOfVertices:
                A,B=AB
                for Ej in Es:           
                    ACL=(A@Q+B@Ej@Y+B@(onp.eye(inputSize)-Ej)@Z)
                    constraints += [                    
                        cp.vstack([cp.hstack([(tau)*Q,      Q*0,ACL.T]),
                                    cp.hstack([Q*0,      onp.eye(stateSize)*(1-tau),Dw]),
                                    cp.hstack([ACL, Dw.T, Q]),
                          ])>>1e-4
                        
                    ]
            prob = cp.Problem(cp.Minimize(-cp.trace(Q)), constraints)
            prob.solve(solver='MOSEK',verbose=True)
            
            print("The optimal value is", prob.value)
            # print("A solution X is")
            P=onp.linalg.inv(Q.value);
            K=(Y.value@P)
            H=Z.value@P 
            for AB in setOfVertices:
                A,B=AB
                print(onp.linalg.eigvals(A+B@K))
                assert(onp.all(onp.abs(onp.linalg.eigvals(A+B@K))<1))
            return P,K,H
        
        P,K,H=synthesizeController()
        
        
        
        
        def costEigPK(x,Q,K,H):
            global mask
            x=onp.reshape(x, (mask.shape[1],1))
            # A,B=approximateOnPointJAC(x,nbrs)
            x=mask@x
            x=x.reshape((1,stateSize+inputSize+1))
            # A,B=approximateOnPointJAC(x,nbrs)
            xState=x[0:1,0:stateSize]
            u=x[0:1,stateSize:stateSize+inputSize]
            p=x[0:1,stateSize+inputSize:]
            # xTT=system.forward(u,xState,p)
            # overallDataset+=[(xState.T,u[0],xTT,computeAB(xState.T,u))]
            v=onp.Inf;
            
            Y=K@Q
            Z=H@Q
            Es=[]         
            
            Dw=onp.eye( stateSize)*0
            combinations=int(2**inputSize)
            for j in range(combinations):
                Es += [onp.diag([int(b) for b in onp.binary_repr(j, inputSize)])]
            
            
            for k in range(paraSize):
                para_v=onp.ones((paraSize,1));
                para_v[k]=p[0:1];
                A,B=computeAB(xState.T,u,para_v)            
                for Ej in Es:           
                    ACL=(A@Q+B@Ej@Y+B@(onp.eye(inputSize)-Ej)@Z)
                    cond =onp.vstack([onp.hstack([(tau)*Q,      Q*0,ACL.T]),
                                    onp.hstack([Q*0,      onp.eye(stateSize)*(1-tau),Dw]),
                                    onp.hstack([ACL, Dw.T, Q]),])
                    # print(constraints[-1].shape)
                    # cond=scipy.linalg.block_diag(constraints)
                    eig=scipy.sparse.linalg.eigsh(cond,1,which='SA',return_eigenvectors=False)
                    # print(eig)
                    v=min(min(onp.real(eig)),v)
                    
            return v
        Q=onp.linalg.inv(P)
        costEig=lambda x: costEigPK(x,Q,K,H)
        b=scipy.optimize.Bounds(mask.T@Bounds.lb,mask.T@Bounds.ub)
        res=scipy.optimize.direct(costEig,bounds=b,locally_biased=True) 
        # res=scipy.optimize.minimize(costEig,res.x,vbounds=Bounds) 
        # halo = HALO(costEig, [[Bounds.lb[i],Bounds.ub[i]] for i in range(0,len(Bounds.lb))], max_feval, max_iter, beta, local_optimizer, verbose)
        # result=halo.minimize();
        result={'best_f':res.fun,'best_x':res.x}
        # break
        print(result['best_f'])
        if result['best_f']>=0:
            break
        else:
            x=result['best_x']
            x=mask@x
            print("last counter example",x)
            x=onp.reshape(x, (1,stateSize+inputSize+1))
            xState=x[0:1,0:stateSize]
            u=x[0:1,stateSize:stateSize+inputSize]
            p=x[0:1,stateSize+inputSize:]
            for k in range(paraSize):
                para_v=onp.ones((paraSize));
                para_v[k]=p[0:1];
                setOfVertices+=[computeAB(xState,u,para_v)]
    return P,K
    
    


#%%

Psat,Ksat=Bemporad()

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

Kinf=-K_Hinf2

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
def MPCsim(x0=None,label="MPC"):
    #%%
    from jax import jit
    horizon=7
    @jit
    def costFun(uG,x0,ref):
        x0S=jnp.reshape(x0*1,(1,stateSize))
        error=jnp.sum(jnp.ravel(uG)**2)/1000
        # horizon=7
        p=jnp.ones((1,paraSize))
        uF=jnp.reshape(uG,(horizon,inputSize))
        for r in range(0,horizon):
            x0SN=system.innerDynamic(x0S,uF[r,:],p)
            error+=jnp.sum(jnp.square(x0SN-ref))
            x0S=x0SN*1
        return error
            
            
        
    story=[]
    if x0 is None:
        xState=onp.zeros((1,stateSize))
        xState[0,0]=0
        xState[0,1]=0
    else:
        xState=onp.reshape(x0,(1,stateSize))
    p=jnp.ones((1,paraSize))
    b=scipy.optimize.Bounds(onp.repeat(Bounds.lb[stateSize:stateSize+inputSize],horizon),
                            onp.repeat(Bounds.ub[stateSize:stateSize+inputSize],horizon))
    for k in range(0,35000):
        ref=onp.array(xState*0)
        ref[0,0]=onp.cos(k/1000)/5+0.5
        ref[0,1]=onp.cos((600+k)/1000)/5
        # err=
        # xState[0,-1]+=-.1
        
        lcost= (lambda u: costFun(u,jnp.array(xState*1),jnp.array(ref)))
        uG=jnp.repeat(jnp.array(Ksat@(xState-ref).T),horizon,1).T.ravel()
        res=scipy.optimize.minimize(lcost,uG,bounds=b,method='SLSQP',options={'maxiter':3})
        # uF=onp.array(K@(xState-ref).T).ravel()
        uF=onp.reshape(res.x[0:inputSize], (inputSize))
        xN=system.innerDynamic(xState,uF,p)
        p=onp.ones((1,paraSize))
        if k>=3000 and k<=4000:
            p[0,2]=.1
        if k>=5000:
            p[0,1]=.1
        else:
            pass
        story+=[(onp.array(xState),onp.array(uF))]
        xState=onp.array(onp.reshape(xN,(1,stateSize)))
        
        print(k,xState)
    
        
    import matplotlib.pyplot as plt
    
    fig, (ax1, ax2)=plt.subplots(1, 2, sharey=False)
    ax1.title.set_text('input '+str(label))
    ax1.plot(onp.array([x[1] for x in story]).reshape((-1,inputSize)))
    
    ax2.title.set_text('state '+str(label))
    ax2.plot(onp.array([x[0] for x in story]).reshape((-1,stateSize)))

import time
tStart=time.time()
MPCsim()   
timeMPC=time.time()-tStart
print(timeMPC)
# #%%
# def Khotare():
#     pass
#     import cvxpy as cp
#     from halo import HALO
    
    
#     max_feval = 8000  # maximum number of function evaluations
#     max_iter = 8000  # maximum number of iterations
#     beta = 1e-2  # beta controls the usage of the local optimizers during the optimization process
#     # With a lower value of beta HALO will use the local search more rarely and viceversa.
#     # The parameter beta must be less than or equal to 1e-2 or greater than equal to 1e-4.
#     local_optimizer = 'L-BFGS-B' # Choice of local optimizer from scipy python library.
#     # The following optimizers are available: 'L-BFGS-B', 'Nelder-Mead', 'TNC' and 'Powell'.
#     # For more infomation about the local optimizers please refer the scipy documentation.
#     verbose = 0  # this controls the verbosity level, fixed to 0 no output of the optimization progress 
#     # will be printed.
    
    
    
#     x=onp.zeros((stateSize))+0.5;
#     u=onp.zeros((inputSize))+0.5;
#     p=onp.ones((paraSize))
#     setOfVertices=[computeAB(x,u,p)]
    
#     while(True):
#         def synthesizeController():
#             Qx=onp.eye(stateSize)/100
#             R=onp.eye(inputSize)/1   #sqrt in realtà
            
#             W=cp.Variable((stateSize,stateSize), symmetric=True)
#             X=cp.Variable((inputSize,inputSize), symmetric=True)
#             Q=cp.Variable((inputSize,stateSize))
#             gamma=cp.Variable((1,1))
            
#             A,B=setOfVertices[0]
#             constraints = [W >> 1*onp.eye(stateSize),
#                             W<<100*onp.eye(stateSize),
#                             # cp.vstack([ cp.hstack([onp.eye(1),      onp.ones((stateSize,1)).T]),
#                             #             cp.hstack([onp.ones((stateSize,1)),W])                               
#                             #   ])>>0
#                            # cp.vstack([ cp.hstack([-X,      (R@Q)]),
#                            #             cp.hstack([(R@Q).T, -W])                               
#                            #   ])<<0
#                            ]
#             L=onp.diag(1/Bounds.ub[:stateSize])
#             for k in range(stateSize):
#                 Lr=L[k:k+1,:]
#                 print(Lr.shape)
#                 constraints+=[                
#                     cp.vstack([cp.hstack([onp.eye(1),   Lr@W]),
#                                 cp.hstack([ W@Lr.T,     W])])>>0
#                     ]
                
#             for AB in setOfVertices:
#                 A,B=AB
#                 constraints += [                    
#                     # cp.vstack([cp.hstack([-W+onp.eye(stateSize),      A@W+B@Q]),
#                     #             cp.hstack([(A@W+B@Q).T,-W])                            
#                     #   ])<<-1e-2
#                     cp.vstack([cp.hstack([W,      (A@W+B@Q).T,W@Qx,Q.T@R]),
#                                 cp.hstack([(A@W+B@Q),W,W*0,Q.T*0]),
#                                 cp.hstack([(W@Qx).T,W*0,gamma*onp.eyMastie(stateSize),Q.T*0]),
#                                 cp.hstack([(Q.T@R).T,Q*0,Q*0,gamma*onp.eye(inputSize)])                           
#                       ])>>1e-2
                    
#                 ]
#             prob = cp.Problem(cp.Minimize(gamma/1000), constraints)
#             prob.solve(solver='MOSEK',verbose=True)
            
#             print("The optimal value is", prob.value)
#             # print("A solution X is")
#             P=onp.linalg.inv(W.value);
#             K=(Q.value@onp.linalg.inv(W.value))
            
#             for AB in setOfVertices:
#                 A,B=AB
#                 print(onp.linalg.eigvals(A+B@K))
#                 assert(onp.all(onp.abs(onp.linalg.eigvals(A+B@K))<1))
#             return P,K
        
#         P,K=synthesizeController()
        
        
        
        
#         def costEigPK(x,P,K):
#             x=onp.reshape(x, (1,stateSize+inputSize+1))
#             # A,B=approximateOnPointJAC(x,nbrs)
#             xState=x[0:1,0:stateSize]
#             u=x[0:1,stateSize:stateSize+inputSize]
#             p=x[0:1,stateSize+inputSize:]
#             # xTT=system.forward(u,xState,p)
#             # overallDataset+=[(xState.T,u[0],xTT,computeAB(xState.T,u))]
#             v=onp.Inf;
#             for k in range(paraSize):
#                 para_v=onp.ones((paraSize,1));
#                 para_v[k]=p[0:1];
#                 A,B=computeAB(xState.T,u,para_v)
#                 eig=scipy.sparse.linalg.eigsh(            
#                     onp.vstack([onp.hstack([P,(A+B@K).T@P]),
#                                 onp.hstack([P@(A+B@K),P])]),1,which='SA',return_eigenvectors=False)
#                 # print(eig)
#                 v=min(min(onp.real(eig)),v)
#             return v
        
#         costEig=lambda x: costEigPK(x,P,K)
        
#         res=scipy.optimize.direct(costEig,bounds=Bounds,locally_biased=True) 
#         # res=scipy.optimize.minimize(costEig,res.x,bounds=Bounds) 
#         # halo = HALO(costEig, [[Bounds.lb[i],Bounds.ub[i]] for i in range(0,len(Bounds.lb))], max_feval, max_iter, beta, local_optimizer, verbose)
#         # result=halo.minimize();
#         result={'best_f':res.fun,'best_x':res.x}
#         print(result['best_f'])
#         if result['best_f']>=0:
#             break
#         else:
#             x=result['best_x']
#             x=onp.reshape(x, (1,stateSize+inputSize+1))
#             xState=x[0:1,0:stateSize]
#             u=x[0:1,stateSize:stateSize+inputSize]
#             p=x[0:1,stateSize+inputSize:]
#             for k in range(paraSize):
#                 para_v=onp.ones((paraSize));
#                 para_v[k]=p[0:1];
#                 setOfVertices+=[computeAB(xState,u,para_v)]
    
#     return K,P
    

