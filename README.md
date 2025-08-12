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



