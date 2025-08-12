#
# This script runs a systematic campaign of different MPC tuning on the AUV (5 states) model.
# It evaluates the tuning based on an integral error metrics, ranks the tuning and suggest the best option. 
#
#
#
#

import jax
from jax import numpy as jnp
from jax import jit
jax.config.update("jax_enable_x64", True)
import sys
sys.dont_write_bytecode=True
import mosek
import os
from functools import partial
import scipy
import numpy as onp
import matplotlib.pyplot as plt
import jax.experimental
import time
import cvxpy as cp
from cvxpy.tests.solver_test_helpers import StandardTestLPs
import numpy.matlib
from matplotlib.ticker import FuncFormatter
from cycler import cycler
from scipy.stats import qmc

# Set the location of the Mosek license file -- update as necessary
os.environ['MOSEKLM_LICENSE_FILE'] = './mosek_license/mosek.lic'
StandardTestLPs.test_lp_0(solver='MOSEK')


plt.rcParams.update({
    "text.usetex": False,
})

plt.rcParams['mathtext.fontset'] = 'stix'
plt.rcParams['font.family'] = 'STIXGeneral'
plt.rcParams['font.size'] = '14'
plt.tight_layout()
plt.show(block=False)


''' 
Classes corresponding to different dynamics and utilities
'''
class BaseBenchmark:
    def __init__(self,noiseCov=None,enableNoise=True):
        self.enableNoise=enableNoise
        if noiseCov is None:
            self.noiseCov=0.1;
        else:
            self.noiseCov=noiseCov;
            print('overwritten noise cov:',noiseCov)
        pass

    def innerDynamic(x0,uT,p,intgralTermRef):
        pass
    
                
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
    def innerDynamic(self,xT, uT,para,intgralTermRef=0):
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
        x5dot=xT[3]-jnp.asarray(intgralTermRef).ravel()[0]
        
        Ts=.01
        xN=jnp.zeros((stateSize,1))        
        xN=xN.at[0].set(xT[0]+Ts*x1dot)
        xN=xN.at[1].set(xT[1]+Ts*x2dot)
        xN=xN.at[2].set(xT[2]+Ts*x3dot)
        xN=xN.at[3].set(xT[3]+Ts*x4dot)
        xN=xN.at[4].set(xT[4]+Ts*x5dot)
        
        return xN.reshape((stateSize,1))


''' 
Parameters to be modified
'''
benchmark_id=5  
b=2  # size of the control validity domain 
total_MPC_tuning_prediction_horizon = 7 # number of different MPC tuning prediction horizon to explore
total_MPC_tuning_gain = 7 # number of different MPC tuning gain to explore
prediction_horizon_min = 10
prediction_horizon_max = 150
gain_min = 0.0001
gain_max = 10
synthesise_new_IS_sat = False # if false use the values provided in Ksat below
Ksat = onp.array([[ -50.98781602,  -48.44657148,  591.40640211,  114.77740694,    5.48530836],
         [  45.53251654,  -45.81310304,  540.44618274,  107.24054371,    5.12516529],
         [  46.94591829,  -47.94972841, -540.43310069, -104.29568136,   -4.98424422],
         [ -41.54824799,  -44.53343709, -492.38032439,  -96.35366138,   -4.60431652]])


switch_dict = {    
    5: lambda: (AUV,5,4,4,scipy.optimize.Bounds(
                            onp.array([-b,-b,-b,-b,-b,-1,-1,-1,-1,0.0]),
                            onp.array([b,b,b,b,b,1,1,1,1,1])),onp.delete(onp.diag([1,1,1,0,0,1,1,1,1,1]),[3,4],axis=0).T),
}
     
systemClass,stateSize,inputSize,paraSize,Bounds,mask=switch_dict.get(benchmark_id)()



system=systemClass(enableNoise=False)
onp.random.seed(0)
jacA=jax.jit(jax.jacobian(system.innerDynamic))
jacB=jax.jit(jax.jacobian(system.innerDynamic,argnums=1))

computeAB=lambda x,u,p: [jacA(x,u,p).reshape((stateSize,stateSize)),jacB(x,u,p).reshape((stateSize,inputSize))]

def generateFault(k,mult):
    p=onp.ones((1,paraSize))
    if k>=4000*mult and k<8000*mult:
        p[0,2]=.1
    elif k>=8000*mult:
        p[0,1]=.1
    else:
        pass
    return p

def generateRef(k,sineTrackMult):
    ref=onp.zeros((1,stateSize))
    ref[0,0]=sineTrackMult*onp.cos(k/1000)/5+0.5
    ref[0,1]=sineTrackMult*onp.cos((600+k)/1000)/5
    return ref


@FuncFormatter
def my_formatter(x, pos):
     return "{}".format(x/100.0)
    
    
def printStory(ax0,ax1,ax2,story,plotError,numStatesToPrint,labelTitle,style,plotLog,printref, haveFault,plotlabel):
    
    
    if plotError:    
        newStory=[]
        for x in story:
            error=x[0].ravel()[0:numStatesToPrint]-x[-1].ravel()[0:numStatesToPrint]
            # print(x)
            # print(error)
            newStory+=[[onp.linalg.norm(error)]+x[1:]]
        # print(newStory)
        story=newStory
        

    mycycler = (cycler('color', ['#1f77b4', '#ff7f0e', '#d62728', '#2ca02c', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf'][0:inputSize]))
    ax1.set_prop_cycle(mycycler)
    # ax1.title.set_text('control signal' )
    ax1.set_ylabel('control signal' )
    
    
    mycycler = (cycler('color', ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf'][0:numStatesToPrint]))
    ax2.set_prop_cycle(mycycler)
    if plotError:
        # ax2.title.set_text('tracking error')
        ax2.set_ylabel('tracking error')
    else:
        # ax2.title.set_text('state')
        ax2.set_ylabel('state')
    # label=None
    
    labelU=[labelTitle if i==0 else None for i in range(0,inputSize)  ]
    labelX=[labelTitle+" - "+"$x_{}(t)$".format(i+1) for i in range(0,numStatesToPrint)]
    labelX=[labelTitle if i==0 else None for i   in range(0,numStatesToPrint)]
    labelR=["ref $x_{}(t)$".format(i+1) for i in range(0,numStatesToPrint)]
    labelR=["ref" if i==0 else None for i   in range(0,numStatesToPrint)]
    
    # ax0.title.set_text('$||\cdot||$ control effort')
    ax0.set_ylabel('$||\cdot||$ control signal')
    ax0.plot([onp.linalg.norm(x[1].ravel()) for x in story],color='#1f77b4',ls=style,label=str(labelTitle))
    # else:
   
    ax1.plot(onp.array([Bounds.lb[stateSize:stateSize+inputSize] for x in story]).reshape((-1,inputSize)),ls='dotted',color='grey',linewidth=1.0)
    ax1.plot(onp.array([Bounds.ub[stateSize:stateSize+inputSize] for x in story]).reshape((-1,inputSize)),ls='dotted',color='grey',linewidth=1.0)
    ax1.plot(onp.array([x[1].ravel() for x in story]).reshape((-1,inputSize)),ls=style,label=labelU) 
    
    if plotLog:
        # story+=[(onp.norm(xState-ref*int(plotError)),uF,p*1,ref)]
        ax2.semilogy(onp.array([x[0].ravel() for x in story]).reshape((-1,1)),ls=style,label=labelTitle)
    else:
        ax2.plot(onp.array([x[0].ravel()[0:numStatesToPrint] for x in story]).reshape((-1,numStatesToPrint)),label=labelX[0:len(story[0])],ls=style)
    
    if printref and not(plotLog):
        ax2.plot(onp.array([x[-1].ravel()[0:numStatesToPrint] for x in story]).reshape((-1,numStatesToPrint)),ls='solid',
                 label=labelR)
        
    mask=[0]+[i  for i in range(1,len(story)) if not(onp.allclose(story[i-1][2],story[i][2]))]+[len(story)]
    color=['white','yellow','greenyellow','yellow','white','yellow','white','yellow']
    print(mask)
    if haveFault:
        for k in range(0,len(mask)-1):
            ax1.axvspan(mask[k],mask[k+1],facecolor=color[k],alpha=.125,ec ='black',zorder=0)
            ax2.axvspan(mask[k],mask[k+1],facecolor=color[k],alpha=.125,ec ='black',zorder=0)
            ax0.axvspan(mask[k],mask[k+1],facecolor=color[k],alpha=.125,ec ='black',zorder=0)
    if plotlabel:
        ax0.legend(loc='lower right')
        # ax1.legend(loc='lower right')
        pass
    ax0.xaxis.set_major_formatter(my_formatter)
    # ax0.set_xlabel('$\mathrm{[seconds]}$')
    ax1.xaxis.set_major_formatter(my_formatter)
    # ax1.set_xlabel('$\mathrm{[seconds]}$')
    ax2.xaxis.set_major_formatter(my_formatter)
    ax2.set_xlabel('$\mathrm{time [seconds]}$')


# Specific function with two subplots defined for plotting and comparing multiple MPC tuning  
def printStoryMultipleMPC(ax3,ax4,story,plotError,numStatesToPrint,labelTitle,style,plotLog,printref, haveFault,plotlabel):
    
    
    if plotError:    
        newStory=[]
        for x in story:
            error=x[0].ravel()[0:numStatesToPrint]-x[-1].ravel()[0:numStatesToPrint]
            # print(x)
            # print(error)
            newStory+=[[onp.linalg.norm(error)]+x[1:]]
        # print(newStory)
        story=newStory
        
    labelU=[labelTitle if i==0 else None for i in range(0,inputSize)  ]
    labelX=[labelTitle+" - "+"$x_{}(t)$".format(i+1) for i in range(0,numStatesToPrint)]
    labelX=[labelTitle if i==0 else None for i   in range(0,numStatesToPrint)]
    labelR=["ref $x_{}(t)$".format(i+1) for i in range(0,numStatesToPrint)]
    labelR=["ref" if i==0 else None for i   in range(0,numStatesToPrint)]

    mycycler = (cycler('color', ['#1f77b4', '#ff7f0e', '#d62728', '#2ca02c', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf'][0:inputSize]))

    # Custom axes for multiple MPC tuning comparison
    ax3.set_ylabel('$||\cdot||$ control signal')
    ax3.set_xlabel('$\mathrm{time [seconds]}$')
    ax3.plot([onp.linalg.norm(x[1].ravel()) for x in story],ls=style,label=str(labelTitle))
    ax3.legend(loc='lower right')

    if plotLog:
        # story+=[(onp.norm(xState-ref*int(plotError)),uF,p*1,ref)]
        ax4.semilogy(onp.array([x[0].ravel() for x in story]).reshape((-1,1)),ls=style,label=labelTitle)
    else:
        ax4.plot(onp.array([x[0].ravel()[0:numStatesToPrint] for x in story]).reshape((-1,numStatesToPrint)),label=labelX[0:len(story[0])],ls=style)
    
    if printref and not(plotLog):
        ax4.plot(onp.array([x[-1].ravel()[0:numStatesToPrint] for x in story]).reshape((-1,numStatesToPrint)),ls='solid',
                 label=labelR)
    ax4.xaxis.set_major_formatter(my_formatter)
    ax4.set_xlabel('$\mathrm{time [seconds]}$')
    if plotError:
        ax4.set_ylabel('tracking error')
    else:
        ax4.set_ylabel('state')

    pass




#%%
tau=1-0.001
def Bemporad():

   
    x=onp.zeros((stateSize))+1;
    u=onp.zeros((inputSize))+1
    p=onp.ones((paraSize))
    setOfVertices=[computeAB(x,u,p)]
    print(setOfVertices)
    trackingNumberOfiterations=0
    while(True):
        
        def synthesizeController():
           
            
            Q=cp.Variable((stateSize,stateSize), symmetric=True)
            Dw=onp.eye(stateSize)*0
            Y=cp.Variable((inputSize,stateSize))
            Z=cp.Variable((inputSize,stateSize))        
            
            
            
            constraints = [Q<<50*onp.eye(stateSize)  ]
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
                if (onp.diag(mask)[k]>0.9):
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
        print("verifier says: ",result['best_f'])
        if result['best_f']>=-1e-9*0:
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
        trackingNumberOfiterations+=1
    print(trackingNumberOfiterations)
    return P,K,len(setOfVertices)
    


#%%
if synthesise_new_IS_sat:
    t_start_synthesis = time.time()
    print("\n Synthesising IS-sat controller ... ")
    Psat,Ksat,numVertPsat=Bemporad()
    synthesis_time = time.time() - t_start_synthesis
    print(f"\n Terminated synthesis of IS-sat controller in {synthesis_time} seconds.\n\n")
    print(Ksat)



#%%
@jit
def simForMC(x0,p,K,lenSim=300):
    xState=jnp.reshape(x0,(1,stateSize))
    p=jnp.reshape(p,(1,paraSize))
    ref=jnp.array(xState*0)
    for k in range(0,lenSim):       
        uF=jnp.clip((K@(xState-ref).T).ravel(),Bounds.lb[stateSize:stateSize+inputSize],Bounds.ub[stateSize:stateSize+inputSize])        
        xN=system.innerDynamic(xState,uF,p)
        xState=jnp.array(jnp.reshape(xN,(1,stateSize)))
        
    return xState
#%%

sampler = qmc.LatinHypercube(d=stateSize+1,seed=1)
sample = sampler.random(n=2500)
sample=qmc.scale(sample, onp.concatenate((onp.ones(1)*0,Bounds.lb[0:stateSize])), onp.concatenate((onp.ones(1),Bounds.ub[0:stateSize])))
def LHSstabilityTest(label,Kgain,threshold=0.1):
    str_comp=['$h_1$','$h_2$','$h_3$']
    for comp in range(0,3):
        story=[]
        if benchmark_id==6:
            for k in range(0,len(sample)):
                x0=onp.resize(sample[k][0:stateSize],(stateSize,1))
                p=onp.ones((paraSize));
                p[comp]=sample[k][stateSize]
                xEnd=simForMC(x0,p,Kgain)
                story+=[(x0,sample[k][stateSize],xEnd)]
                # print('.',end='')
        
        contracted=[x for x in story if onp.linalg.norm(x[-1])<threshold*onp.linalg.norm(x[0])]
        fig = plt.figure()
        
        ax = fig.add_subplot(projection='3d')
        for xp in contracted:    
            ax.scatter(xp[0][0], xp[0][1], xp[1], facecolor="gold")
        notContracted=[x for x in story if onp.linalg.norm(x[-1])>=threshold*onp.linalg.norm(x[0])]
        for xp in notContracted:    
            ax.scatter(xp[0][0], xp[0][1], xp[1], facecolor="blue")
        plt.title(str_comp[comp]+" "+label)
        print(str_comp[comp]+" "+label,len(contracted),len(notContracted))
    ax.set_xlabel('$x_1$')
    ax.set_ylabel('$x_2$')
    ax.set_zlabel('$\\rho$')

if False:
    LHSstabilityTest("$K_\mathrm{IS-sat}$",Ksat)
    LHSstabilityTest("$\mathcal{H}_\infty^a$",-K_Hinf2)


def simulateController(K,labelTitle,ax0,ax1,ax2,x0=None,printref=True,style='-',onlySim=False,plotLog=False,integralTermToTrack=0.2,
                       plotlabel=False,plotError=False,numStatesToPrint=stateSize,haveFault=True,sineTrack=False,mult=1,simTime=12000):
    story=[]
    if not callable(K):
        def computeU(xState,ref):
            uF=onp.clip((K@(xState-ref).T).ravel(),Bounds.lb[stateSize:stateSize+inputSize],Bounds.ub[stateSize:stateSize+inputSize])
            return uF
    else:
        computeU=K
        
    if x0 is None:
        xState=onp.zeros((1,stateSize))
        # xState[0,0]=-0.3
        # xState[0,1]=0.2
    else:
        xState=onp.reshape(x0,(1,stateSize))
    p=onp.ones((1,paraSize))
    tStart=time.time()
    sineTrackMult=0
    if sineTrack:
        sineTrackMult=1
    for k in range(0,int(simTime*mult)):

        ref=generateRef(k,sineTrackMult)
        # err=
        # xState[0,-1]+=-.1
        # uF=onp.clip((K@(xState-ref).T).ravel(),Bounds.lb[stateSize:stateSize+inputSize],Bounds.ub[stateSize:stateSize+inputSize])
        # uF=onp.array(K@(xState-ref).T).ravel()
        uF=computeU(xState,ref)
        xN=system.innerDynamic(xState,uF,p,integralTermToTrack)
        p=onp.ones((1,paraSize))
        if haveFault:
            p=generateFault(k,mult)
        
        if integralTermToTrack>1e-4:
            ref[0,-2]=integralTermToTrack*1
            
        
        story+=[[xState,uF,p*1,ref]]
        xState=onp.array(onp.reshape(xN,(1,stateSize)))
        
        # print(xState)
    if onlySim:
        return xState,story
    timeStaticFeedback=time.time()-tStart
    printStory(ax0,ax1,ax2,story,plotError,numStatesToPrint,labelTitle,style,plotLog,printref, haveFault,plotlabel)
    return timeStaticFeedback



def simulateControllerMultipleMPC(K,labelTitle,ax3,ax4,x0=None,printref=True,style='-',onlySim=False,plotLog=False,integralTermToTrack=0.2,
                       plotlabel=False,plotError=False,numStatesToPrint=stateSize,haveFault=True,sineTrack=False,mult=1):
    story=[]
    if not callable(K):
        def computeU(xState,ref):
            uF=onp.clip((K@(xState-ref).T).ravel(),Bounds.lb[stateSize:stateSize+inputSize],Bounds.ub[stateSize:stateSize+inputSize])
            return uF
    else:
        computeU=K
        
    if x0 is None:
        xState=onp.zeros((1,stateSize))
        # xState[0,0]=-0.3
        # xState[0,1]=0.2
    else:
        xState=onp.reshape(x0,(1,stateSize))
    p=onp.ones((1,paraSize))
    tStart=time.time()
    sineTrackMult=0
    if sineTrack:
        sineTrackMult=1
    for k in range(0,int(12000*mult)):

        ref=generateRef(k,sineTrackMult)
        # err=
        # xState[0,-1]+=-.1
        # uF=onp.clip((K@(xState-ref).T).ravel(),Bounds.lb[stateSize:stateSize+inputSize],Bounds.ub[stateSize:stateSize+inputSize])
        # uF=onp.array(K@(xState-ref).T).ravel()
        uF=computeU(xState,ref)
        xN=system.innerDynamic(xState,uF,p,integralTermToTrack)
        p=onp.ones((1,paraSize))
        if haveFault:
            p=generateFault(k,mult)
        
        if integralTermToTrack>1e-4:
            ref[0,-2]=integralTermToTrack*1
            
        
        story+=[[xState,uF,p*1,ref]]
        xState=onp.array(onp.reshape(xN,(1,stateSize)))
        
        # print(xState)
    if onlySim:
        return xState,story
    

    if plotError:
        # Calc and plot the integral of the absolute error
        error_norms = [onp.linalg.norm(x[0].ravel()[0:numStatesToPrint] -
                                       x[-1].ravel()[0:numStatesToPrint])
                       for x in story]
        integral_error = onp.sum(onp.abs(error_norms)) 
    
    else:
        integral_error = None  

    timeStaticFeedback=time.time()-tStart
    printStoryMultipleMPC(ax3,ax4,story,plotError,numStatesToPrint,labelTitle,style,plotLog,printref, haveFault,plotlabel)

    return timeStaticFeedback, integral_error
# plt.plot(onp.array([x[0].T@P@x[0] for x in story]).reshape((-1,stateSize)))
# plt.figure()




def genMPC():
    
    horizon=56
    gain = 0.0001

    print("Type of horizon before reshape:", type(horizon), horizon)
    print("Type of gain before reshape:", type(gain), gain)


    @jit
    def costFun(uG,x0,ref):
        x0S=jnp.reshape(x0*1,(1,stateSize))
        error=0
        # horizon=7
        p=jnp.ones((1,paraSize))
        uF=jnp.reshape(uG,(horizon,inputSize))
        for r in range(0,horizon):
            x0SN=system.innerDynamic(x0S,uF[r,:],p)
            error+=jnp.sum(jnp.square(x0SN.reshape((1,stateSize))-ref.reshape((1,stateSize))))
            error+=jnp.sum(jnp.ravel(uF[r,:])**2)/gain
            x0S=x0SN*1
        return error

    b=scipy.optimize.Bounds(onp.repeat(Bounds.lb[stateSize:stateSize+inputSize],horizon),
                            onp.repeat(Bounds.ub[stateSize:stateSize+inputSize],horizon))
    conGra=jax.jit(jax.grad(costFun))
    
    
    def computeUMPC(xState,ref):
        lcost= lambda u: (costFun(u,jnp.array(xState*1),jnp.array(ref)),conGra(u,jnp.array(xState*1),jnp.array(ref)))
        uG=jnp.repeat(jnp.array(Ksat@(xState-ref).T),horizon,1).T.ravel()
        res=scipy.optimize.minimize(lcost,uG,bounds=b,jac=True, method='L-BFGS-B')
        # uF=onp.array(K@(xState-ref).T).ravel()
        uF=onp.reshape(res.x,(horizon,inputSize))
        uF=uF[0,:]*1
        return uF
    return computeUMPC


                
def genMPCMultipleTuning(horizon, gain):
    # testing different MPC tuning
    
    @jit
    def costFun(uG,x0,ref):
        x0S=jnp.reshape(x0*1,(1,stateSize))
        error=0
        # horizon=7
        p=jnp.ones((1,paraSize))
        uF=jnp.reshape(uG,(int(horizon),inputSize))
        for r in range(0,int(horizon)):
            x0SN=system.innerDynamic(x0S,uF[r,:],p)
            error+=jnp.sum(jnp.square(x0SN.reshape((1,stateSize))-ref.reshape((1,stateSize)))) # this could be multiplied by Q
            error+=jnp.sum(jnp.ravel(uF[r,:])**2)*gain
            x0S=x0SN*1


        return error
                
    b=scipy.optimize.Bounds(onp.repeat(Bounds.lb[stateSize:stateSize+inputSize],horizon),
                            onp.repeat(Bounds.ub[stateSize:stateSize+inputSize],horizon))
    conGra=jax.jit(jax.grad(costFun))
    
    
    def computeUMPC(xState,ref):
        lcost= lambda u: (costFun(u,jnp.array(xState*1),jnp.array(ref)),conGra(u,jnp.array(xState*1),jnp.array(ref)))
        uG=jnp.repeat(jnp.array(Ksat@(xState-ref).T),horizon,1).T.ravel()
        res=scipy.optimize.minimize(lcost,uG,bounds=b,jac=True, method='L-BFGS-B')
        # uF=onp.array(K@(xState-ref).T).ravel()
        uF=onp.reshape(res.x,(int(horizon),int(inputSize)))
        uF=uF[0,:]*1
        return uF
    return computeUMPC



if benchmark_id==5:

    print("\nComparing multiple MPC tuning")
    
    prediction_horizon_vector = onp.linspace(prediction_horizon_min, prediction_horizon_max, num=total_MPC_tuning_prediction_horizon).astype(int)
    gain_vector = onp.logspace(onp.log10(gain_min), onp.log10(gain_max), num=total_MPC_tuning_gain)

    #horizon=50

    #print("Type of horizon before reshape:", type(horizon), horizon)
    #print("Type of gain before reshape:", type(gain), gain)


    integral_error_results_mpc_tuning = []
    names_mpc_tuning = []
    prediction_horizon_mpc_tuning = []
    gain_mpc_tuning = []
    time_synthesis_mpc_tuning = []
    time_simulation_mpc_tuning = []

    fig, (ax3,ax4)=plt.subplots(2, 1, sharey=False,dpi=160,gridspec_kw={'height_ratios': [3, 3]})
    #fig.set_size_inches(6, 10) 
    for i_tuning_prediction_horizon in range(total_MPC_tuning_prediction_horizon):
        for i_tuning_gain in range(total_MPC_tuning_gain): 

            print("\nMPC tuning prediction horizon number: #" + str(i_tuning_prediction_horizon+1) + "/" + str(total_MPC_tuning_prediction_horizon))
            print("MPC tuning gain number: #" + str(i_tuning_gain+1) + "/" + str(total_MPC_tuning_gain))
            horizon=prediction_horizon_vector[i_tuning_prediction_horizon]
            gain=gain_vector[i_tuning_gain]
            print("MPC tuning prediction horizon: " + str(horizon))
            print("MPC tuning gain: " + str(gain))

            t_start_synthesis_mpc = time.time()           
            computeUMPC=genMPCMultipleTuning(horizon, gain)
            synthesis_time = time.time() - t_start_synthesis_mpc

            name_controller = 'MPC_ph' + str(horizon) + '_g' + str(gain)
            t_start_simulation_mpc = time.time()           
            _, integral_error = simulateControllerMultipleMPC(computeUMPC,name_controller,ax3,ax4, printref=False,numStatesToPrint=stateSize-1,haveFault=False,plotlabel=True,style='dashed',sineTrack=True,x0=onp.ones((1,stateSize))+2,plotError=True,plotLog=True)
            simulation_time = time.time() - t_start_simulation_mpc

            # Saving performance index (integral error)
            print("Integral error for " + str(name_controller) + "= " + str(integral_error))
            integral_error_results_mpc_tuning.append(integral_error)
            names_mpc_tuning.append(name_controller)
            prediction_horizon_mpc_tuning.append(horizon)
            gain_mpc_tuning.append(gain)
            time_synthesis_mpc_tuning.append(synthesis_time)
            time_simulation_mpc_tuning.append(simulation_time)

    plt.tight_layout()
    plt.show(block=False)

    plt.figure()
    plt.bar(range(len(integral_error_results_mpc_tuning)), integral_error_results_mpc_tuning)
    plt.xticks(range(len(names_mpc_tuning)), names_mpc_tuning, rotation=90)
    plt.ylabel("Perfformance index (lower is better)")
    plt.title("MPC tuning comparison")
    plt.tight_layout()
    plt.show(block=False)


    # Discarding unstable tuning 
    mask_unstable_tuning = ~onp.isnan(integral_error_results_mpc_tuning)
    stable_integral_error_results_mpc_tuning = [x for x, m in zip(integral_error_results_mpc_tuning, mask_unstable_tuning) if m]
    stable_names_mpc_tuning = [n for n, m in zip(names_mpc_tuning, mask_unstable_tuning) if m]
    stable_prediction_horizon_mpc_tuning = [n for n, m in zip(prediction_horizon_mpc_tuning, mask_unstable_tuning) if m]
    stable_gain_mpc_tuning = [n for n, m in zip(gain_mpc_tuning, mask_unstable_tuning) if m]
    stable_time_synthesis_mpc_tuning = [n for n, m in zip(time_synthesis_mpc_tuning, mask_unstable_tuning) if m]
    stable_time_simulation_mpc_tuning = [n for n, m in zip(time_simulation_mpc_tuning, mask_unstable_tuning) if m]

    tuning_pairs = zip(stable_integral_error_results_mpc_tuning, stable_names_mpc_tuning)
    tuning_pairs_sorted = sorted(tuning_pairs)
    stable_integral_error_results_mpc_tuning_sorted, stable_names_mpc_tuning_sorted = zip(*tuning_pairs_sorted)

    # Performance analysis
    plt.figure(dpi=300)
    plt.bar(range(len(stable_integral_error_results_mpc_tuning_sorted)), stable_integral_error_results_mpc_tuning_sorted)
    plt.xticks(range(len(stable_names_mpc_tuning_sorted)), stable_names_mpc_tuning_sorted, rotation=90)
    plt.ylabel("Performance index (lower is better)")
    plt.title("MPC tuning comparison sorted")
    plt.tight_layout()
    plt.show(block=False)

    # Time analysis
    plt.figure(dpi=300)
    plt.bar(range(len(time_simulation_mpc_tuning)), time_simulation_mpc_tuning)
    plt.xticks(range(len(names_mpc_tuning)), names_mpc_tuning, rotation=90)
    plt.axhline(y=120, color='red', linestyle='--', label='real time')
    plt.ylabel("time [s]")
    plt.title("MPC simulation time")
    plt.tight_layout()
    plt.show(block=False)

    plt.figure(dpi=300)
    plt.bar(range(len(time_synthesis_mpc_tuning)), time_synthesis_mpc_tuning)
    plt.xticks(range(len(names_mpc_tuning)), names_mpc_tuning, rotation=90)
    plt.ylabel("time [s]")
    plt.title("MPC synthesis time")
    plt.tight_layout()
    plt.show(block=False)

    time_simulation_slow = [x for x in time_simulation_mpc_tuning if float(x) >= 120]
    number_simulations_too_slow = len(time_simulation_slow)/len(time_simulation_mpc_tuning)*100

    print("Tuning comparison terminated")


    # Extracting best tuning
    index_best_mpc_tuning = integral_error_results_mpc_tuning.index(min(integral_error_results_mpc_tuning))
    print(f"The best MPC obtained is = {names_mpc_tuning[index_best_mpc_tuning]}")
    print(f"Best MPC obtained with prediction horizon = {prediction_horizon_mpc_tuning[index_best_mpc_tuning]} and gain = {gain_mpc_tuning[index_best_mpc_tuning]}")
    print(f"The best MPC was synthesised in {time_synthesis_mpc_tuning[index_best_mpc_tuning]} s")
    print(f"The best MPC was simulated in {time_simulation_mpc_tuning[index_best_mpc_tuning]} s")

    print(f"\nOn average, the MPC tuning were synthesised in {onp.mean(stable_time_synthesis_mpc_tuning)} +- {onp.std(stable_time_synthesis_mpc_tuning)} s")
    print(f"On average, the MPC tuning were simulated in {onp.mean(stable_time_simulation_mpc_tuning)} +- {onp.std(stable_time_simulation_mpc_tuning)} s")
    print(f"{number_simulations_too_slow} of the MPC simulations are slower than real time (equating to {len(time_simulation_slow)} simulations).")

    print("\nComparing Is-sat and selected MPC ... ")

    fig, (ax0,ax1, ax2)=plt.subplots(3, 1, sharey=False,dpi=160,gridspec_kw={'height_ratios': [2, 3, 3]})
    fig.set_size_inches(6, 10) 

    # simulating best MPC tuning
    computeUMPC=genMPCMultipleTuning(prediction_horizon_mpc_tuning[index_best_mpc_tuning], gain_mpc_tuning[index_best_mpc_tuning])
    name_best_controller = 'MPC_ph' + str(prediction_horizon_mpc_tuning[index_best_mpc_tuning]) + '_' + str(gain_mpc_tuning[index_best_mpc_tuning])
    #timeMPC, integral_error = simulateControllerMultipleMPC(computeUMPC,'MPC',ax3,ax4, printref=False,numStatesToPrint=stateSize-1,haveFault=True,plotlabel=True,style='dashed',sineTrack=True,x0=onp.ones((1,stateSize))+2,plotError=True,plotLog=True)

    timeMPC=simulateController(computeUMPC,'MPC',ax0,ax1,ax2,
                       printref=False,numStatesToPrint=stateSize-1,haveFault=True,plotlabel=True,style='dashed',sineTrack=True,x0=onp.ones((1,stateSize))+2,plotError=True,plotLog=True,simTime=12000)
    

    timeStaticFeedback=simulateController(Ksat,'$K_\mathrm{IS-sat}$',ax0,ax1,ax2,
                       printref=False,numStatesToPrint=stateSize-1,haveFault=True,plotlabel=True,style='solid',sineTrack=True,x0=onp.ones((1,stateSize))+2,plotError=True,plotLog=True,simTime=12000)
    plt.tight_layout()
    ax2.legend(loc="lower left")
    plt.show(block=True)

    #%%
    print("Final comparison: time MPC: {} --- time static: {}".format(timeMPC,timeStaticFeedback))
