## Installation

The code was tested on OS: Linux Ubuntu 20.04  
Three methods are hereby provided to install the following main dependencies.
    
First, clone the repository:
```
git clone https://github.com/AndreaPuffo/uncertain_cegis.git
cd uncertain_cegis
```

  
### 1.1) Approach 1: install the requirements at system level (not recommended)
If Python3.12 is available on your machine, you can install the required packages at system level with:
```  
pip3 install cvxpy
pip3 install mosek
pip3 install -r ./requirements.txt  
```

For completing Mosek installation, refer to: https://docs.mosek.com/latest/install/installation.html
In short:
```  
cd /home/$whoami (TODO:fix)
cd anaconda3/envs/env_unc_cegis/lib/python3.12/site-packages/mosek/10.2/tools/platform/<PLATFORM>/bin
```
  
"where <PLATFORM> is linux64x86 or linuxaarch64 depending on the version of MOSEK installed." 
   
TODO: add installation of jax aside (it depends on whether you have a CPU or GPU).  
   
Move to step 2).


### 1.2) Approach 2: clone the Anaconda environment
If [Anaconda](https://docs.anaconda.com/free/anaconda/install/) is installed on your system, you can clone the environment with: 

```
conda env create -f documentation/environment_unc_cegis.yml
conda activate env_unc_cegis
```

TODO: environment file not including Mosek, to be updated.
pip3 install mosek  
  
  
TODO: add installation of jax aside (it depends on whether you have a CPU or GPU).  
   
(use `conda deactivate` upon completion.)

Move to step 2).



### 1.3) Approach 3: create a Python virtual environment
  
If Python3.12 is installed on your system, the code can be run in a [virtual environment](https://docs.python.org/3/library/venv.html). Start as follows:
```
pip3 install virtualenv
python3 -m venv unc_cegis_venv
source unc_cegis_venv/bin/activate
python -V
pip3 install -r documentation/requirements.txt  
```
    
TODO: add installation of jax aside (it depends on whether you have a CPU or GPU).  
   
TODO: requirement file not including Mosek, to be updated.
   
(use `deactivate` upon completion.)

Move to step 2).


### 3) Test ANLC successful installation
These commands will run the training for the *nonlinear AUV* system:
```
cd uncertain_cegis/code
python3 main_poly_auv_disjunction.py  
```
  
TODO: update new main.    
   
