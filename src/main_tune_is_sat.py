#
# This script runs a systematic campaign of different IS-sat tuning on the AUV (5 states) model.
# 
#
# date: 18/08/2025
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
total_issat_epsilon = 10 # number of different epsilon values to explore
total_issat_eta = 10 # number of different eta values to explore
epsilon_min = 0.0001
epsilon_max = 0.01
eta_min = 5
eta_max = 5000
verbose_IS_sat=False  # enable if interested in monitoring the status of the synthesis

tau = 1-0.001


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




#%%
def SynthesiseISsat(verbose_IS_sat, eta, epsilon):


    x=onp.zeros((stateSize))+1;
    u=onp.zeros((inputSize))+1
    p=onp.ones((paraSize))
    setOfVertices=[computeAB(x,u,p)]
    print(setOfVertices)
    trackingNumberOfiterations=0
    errorSynthesis = False


    while(True):
        
        def synthesizeController():
           
            
            Q=cp.Variable((stateSize,stateSize), symmetric=True)
            Dw=onp.eye(stateSize)*0
            Y=cp.Variable((inputSize,stateSize))
            Z=cp.Variable((inputSize,stateSize))        
            
            
            
            constraints = [Q<<eta*onp.eye(stateSize)  ]
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
                          ])>>epsilon
                        
                    ]
            prob = cp.Problem(cp.Minimize(-cp.trace(Q)), constraints)
            prob.solve(solver='MOSEK',verbose=False)
            
            if verbose_IS_sat:
                print("The optimal value is", prob.value)
            # print("A solution X is")
            try:
                P=onp.linalg.inv(Q.value);
            except:
                print("Problem with Q!")
                errorSynthesis = True
                return None, None, None
                
                
            K=(Y.value@P)
            H=Z.value@P 
            for AB in setOfVertices:
                A,B=AB
                if verbose_IS_sat:
                    print("Eigenvalues A+BK = ")
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
        
        try:
            Q=onp.linalg.inv(P)
        except:
            print("Problem with P!")
            errorSynthesis = True
            return None, None, None, None, errorSynthesis

        costEig=lambda x: costEigPK(x,Q,K,H)
        b=scipy.optimize.Bounds(mask.T@Bounds.lb,mask.T@Bounds.ub)
        # res=scipy.optimize.direct(costEig,bounds=b,locally_biased=True) 
        res=scipy.optimize.shgo(costEig,bounds=b,options={"f_tol":1e-6}) 
        # res=scipy.optimize.minimize(costEig,res.x,bounds=Bounds) 
        # halo = HALO(costEig, [[Bounds.lb[i],Bounds.ub[i]] for i in range(0,len(Bounds.lb))], max_feval, max_iter, beta, local_optimizer, verbose)
        # result=halo.minimize();
        result={'best_f':res.fun,'best_x':res.x}
        if verbose_IS_sat:
            print("verifier says: ",result['best_f'])
        if result['best_f']>=-1e-9*0:
            print("Number of iterations: ")
            print(trackingNumberOfiterations)
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
        print(f"Iteration #{trackingNumberOfiterations} completed.\n")
        
    return P, K, len(setOfVertices), trackingNumberOfiterations, errorSynthesis 
    
    


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


if benchmark_id==5:

    print("\nComparing multiple IS-sat tuning")
    eps_vector = onp.linspace(epsilon_min, epsilon_max, num=total_issat_epsilon)
    eta_vector = onp.linspace(eta_min, eta_max, num=total_issat_eta)

    KSat_history = []
    time_synthesis = []
    time_synthesis_issat = []
    time_simulation_issat = []
    success_history = []
    eps_success_history = []
    eta_success_history = []
    name_history = []
    no_iteration_history = []


    fig, (ax3,ax4)=plt.subplots(2, 1, sharey=False,dpi=160,gridspec_kw={'height_ratios': [3, 3]})
    #fig.set_size_inches(6, 10) 
    for i_tuning_eps in range(total_issat_epsilon):
        for i_tuning_eta in range(total_issat_eta): 

            epsilon=eps_vector[i_tuning_eps]
            eta=eta_vector[i_tuning_eta]

            #epsilon = 1e-6 
            #eta = 500

            print("\nIs-sat number: #" + str(i_tuning_eps+1) + "/" + str(total_issat_epsilon))
            print("IS-sat eta number: #" + str(i_tuning_eta+1) + "/" + str(total_issat_eta))
            print(f"epsilon = {epsilon}") 
            print(f"eta = {eta}") 

            if (eta>epsilon and epsilon>0.0):

                name_controller = 'IS-sat_eps' + str(epsilon) + '_eta' + str(eta)

                t_start_synthesis_mpc = time.time()   
                t_start_synthesis = time.time()
                print("\n Synthesising IS-sat controller ... ")

                Psat, Ksat, numVertPsat, numIterations, errorSynthesis = SynthesiseISsat(verbose_IS_sat, eta, epsilon)
                synthesis_time = time.time() - t_start_synthesis
                print(f"\n Terminated synthesis of IS-sat controller in {synthesis_time} seconds.\n\n")

                if not errorSynthesis:
                    print("Ksat = ")
                    print(Ksat)
                    synthesis_time = time.time() - t_start_synthesis_mpc
                    
                    success_history.append(True)
                    KSat_history.append(Ksat)
                    time_synthesis_issat.append(synthesis_time)
                    eps_success_history.append(epsilon)
                    eta_success_history.append(eta)
                    name_history.append(name_controller)
                    no_iteration_history.append(numIterations)

                else: 
                    print("Error during synthesis!")
                    synthesis_time = float("nan")
                    
                    success_history.append(False)
                    KSat_history.append(float("nan"))
                    time_synthesis_issat.append(synthesis_time)
                    eps_success_history.append(epsilon)
                    eta_success_history.append(eta)
                    name_history.append(name_controller)
                    no_iteration_history.append(numIterations)

            else: 
                print("Combination of epsilon and eta not valid")
                success_history.append(False)
                KSat_history.append(float("nan"))
                time_synthesis_issat.append(synthesis_time)
                eps_success_history.append(epsilon)
                eta_success_history.append(eta)
                name_history.append(name_controller)
                no_iteration_history.append(numIterations)

    # Simulation of the successful trained control laws
    fig, (ax0,ax1, ax2)=plt.subplots(3, 1, sharey=False,dpi=160,gridspec_kw={'height_ratios': [2, 3, 3]})
    fig.set_size_inches(6, 10) 

    for iSim in range(success_history.__len__()):

        if success_history[iSim] == True:
            # simulate the stable control laws 
            simulation_time=simulateController(KSat_history[iSim], name_history[iSim],ax0,ax1,ax2,
                            printref=False,numStatesToPrint=stateSize-1,haveFault=True,plotlabel=True,style='solid',sineTrack=True,x0=onp.ones((1,stateSize))+2,plotError=True,plotLog=True,simTime=12000)
    plt.tight_layout()
    ax2.legend(loc="lower left")
    plt.show(block=True)


    '''
    plt.figure()
    plt.bar(range(len(integral_error_results_mpc_tuning)), integral_error_results_mpc_tuning)
    plt.xticks(range(len(names_mpc_tuning)), names_mpc_tuning, rotation=90)
    plt.ylabel("Perfformance index (lower is better)")
    plt.title("MPC tuning comparison")
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
'''



