## Installation

The code was tested on OS: Linux Ubuntu 20.04  
Three methods are hereby provided to install the following main dependencies.
    
First, clone the repository:
```
git clone https://github.com/AndreaPuffo/uncertain_cegis.git
cd uncertain_cegis
```

  
### 1.1) Approach 1: install the requirements at system level (not recommended)
If Python3.11 is available on your machine, you can install the required packages at system level with:
```  
pip3 install -r ./requirements.txt  
```

For completing Mosek installation, refer to: https://docs.mosek.com/latest/install/installation.html
   
After having activated a valid Mosek license, copy the license file `mosek.lic` (do not rename the file) inside the code folder as: `uncertain_cegis/src/mosek_license/mosek.lic`.  
  
Installation of `jax`: the following instructions are tested on a system relying on CPU only. If you have a GPU, `jax` might require an alternative version. TODO: test on a GPU.      
   
Move to step 2).


### 1.2) Approach 2: clone the Anaconda environment
If [Anaconda](https://docs.anaconda.com/free/anaconda/install/) is installed on your system, you can clone the environment with: 

```
conda env create -f documentation/env_is_sat.yml
conda activate env_is_sat
```
  
Next, you need to activate a Mosek license (refer to: https://docs.mosek.com/latest/install/installation.html).  
After having activated a valid license, copy the license file `mosek.lic` (do not rename the file) inside the code folder as: `uncertain_cegis/src/mosek_license/mosek.lic`.       
  
(use `conda deactivate` upon completion.)

Move to step 2).



### 1.3) Approach 3: create a Python virtual environment
  
If Python3.9 is installed on your system, the code can be run in a [virtual environment](https://docs.python.org/3/library/venv.html). Start as follows:
```
pip3 install virtualenv
python3.9 -m venv venv_is_sat
source venv_is_sat/bin/activate
python -V
pip3 install -r documentation/requirements.txt  
```
    
Next, you need to activate a Mosek license (refer to: https://docs.mosek.com/latest/install/installation.html).  
After having activated a valid license, copy the license file `mosek.lic` (do not rename the file) inside the code folder as: `uncertain_cegis/src/mosek_license/mosek.lic`.  
   
(use `deactivate` upon completion.)

Move to step 2).


### 2) Test IS-sat successful installation
These commands will run the training for the *nonlinear AUV* system:
```
cd uncertain_cegis/src
python3 main_train_is_sat.py  
```
    
