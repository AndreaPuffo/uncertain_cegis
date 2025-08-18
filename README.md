# IS-sat: a CEGIS-based approach to synthesise robust control functions for uncertain systems affected by actuator faults using LMIs
This repository contains the code for the paper:  
**Fault-tolerant control of nonlinear systems: An inductive synthesis approach**

  
## Scope of the code
This software framework **automatically** synthesises:  
1. a **stabilising controller** for a desired equilibrium of a linear (or linearised) system;  
2. a **formal verification** of the closed-loop stability.
  
  
## Step-by-step installation instructions  
1. Instructions on installation are available within the ![INSTALLATION](./documentation/INSTALLATION.md/) file.    
    
2. Is-sat runs on `Python 3.11`, `jax 0.4` and `mosek 10.2.` .  
We provide installation instructions which are not OS-dependent, such that the code can be run on diverse computational platforms.   
The results were generate and verified on the following OSs:   

|  | OS |
| :---:   | :---: |
| Linux Ubuntu 20.04 |  :white_check_mark:  |
| Linux Ubuntu 24.04 |  :white_check_mark:  |


## Overview of the code
1. To reproduce the results of the paper, synthesise a new IS-sat control law for the provided examples of for your own system, check this ![script](./src/main_train_is_sat.py).  
  
2. To reproduce the nonlinear MPC tuning of the paper, or to tune your own nonlinear MPC control law, check this ![script](./src/main_tune_mpc.py).  
  
3. To explore different IS-sat tuning, check this ![script](./src/main_tune_is_sat.py).  
