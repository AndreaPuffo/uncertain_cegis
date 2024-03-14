#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat May  6 13:39:58 2023

@authors: Davide Grande
          Andrea Peruffo

A function collecting the parameters of the training.

"""

import numpy as np
import torch


def set_params():

    campaign_params = {
        'init_seed': 5,        # initial campaign seed
        'campaign_run': 400084,  # number of the run.
                               # The results will be saved in /results/campaign_'campaign_run'
        'tot_runs': 10,        # total number of runs of the campaigns (each one with a different seed)
        'max_loop_number': 1,  # number of loops per run (>1 means that the weights will be re-initialised).
                                # default value = 1.
        'max_iters': 2000,     # number of maximum learning iterations per run
        'system_name': "2d_auv_faulty",  # name of the systems to be controlled
        'x_star': torch.tensor([0.5, 0.0]),  # target equilibrium point
    }

    # Parameters for learner
    learner_params = {
        'N': 500,  # initial dataset size
        'N_max': 1000,  # maximum dataset size (if using a sliding window)
        'sliding_window': True,  # use sliding window
        'learning_rate': 0.01,  # learning rate Lyapunov branch
        'learning_rate_c': 0.01,  # learning rate control branch
        'use_scheduler': True,
        # use LR scheduler to allow dynamic learning rate reducing based on some validation measurements
        'sched_T': 500,  # cosine annealing scheduler period
        'print_interval': 200,  # interval of loss function printouts
        'enforce_CLF_shape': True,  # require the CLF to approximate a desired shape, i.e. enforce --> L_ELR < tau_overbar
        'tau_overbar': 0.1,  # maximum error on L_ELR
    }

    # Parameters for Lyapunov ANN
    lyap_params = {
        'n_input': 2, # input dimension (n = n-dimensional system)
        'beta_sfpl': 2,  # the higher, the steeper the Softplus, the better approx. sfpl(0) ~= 0
        'clipping_V': True,  # clip weight of Lyapunov ANN
        'size_layers': [10, 10, 1],  # CAVEAT: the last entry needs to be = 1 (this ANN outputs a scalar)!
        'lyap_activations': ['pow2', 'linear', 'linear'],
        'lyap_bias': [False, False, False],
    }

    # Parameters for control ANN
    control_params = {
        'use_lin_ctr': False,  # use linear control law  -- defined as 'phi' in the ANLC publication
        'lin_contr_bias': True,  # use bias on linear control layer
        'control_initialised': False,  # initialised control ANN with pre-computed LQR law
        'init_control': torch.tensor([[2.20381465e+02,  1.87398469e+03],
                                      [2.20381465e+02, -1.87398469e+03],
                                      [6.98044794e-13, -1.46778796e+03]]),  # initial control solution
        'size_ctrl_layers': [30, 3],  # CAVEAT: the last entry is the number of control actions!
        'ctrl_bias': [True, True],
        'ctrl_activations': ['tanh', 'linear'],
        'use_saturation': True,
        'ctrl_sat': [37.1, 37.1, 37.1],  # this vector needs to be as long as 'size_ctrl_layers[-1]' (same size as the control vector)
    }

    falsifier_params = {
        # a) SMT parameters
        'gamma_underbar': 0.1,  # domain lower boundary
        'gamma_overbar': 1.0,   # domain upper boundary
        'zeta_SMT': 200,  # how many points are added to the dataset after a CE box
                          # is found
        'epsilon': 0.0,   # parameters to further relax the SMT check on the Lie derivative conditions.
                          # default value = 0 (inspect utilities/Functions/CheckLyapunov for further info).
                          
        # b) Discrete Falsifier parameters
        'grid_points': 60,  # sampling size grid
        'zeta_D': 100,  # how many points are added at each DF callback
        'use_elliptical_domain': False,
        'inner_domain': [0.1, 0.1],  # inner value of the elliptical domain
    }


    loss_function = {
        # Loss function tuning
        'alpha_1': 1.0,  # weight V
        'alpha_2': 1.0,  # weight V_dot
        'alpha_3': 1.0,  # weight V0
        'alpha_4': 1.0,  # weight tuning term V
        'alpha_roa': 0.001*falsifier_params['gamma_overbar'],  # Lyapunov function steepness
        'alpha_5': 1.0,  # general scaling factor
        'off_Lie': 0.0,   # additional penalisation of the Lie derivative
        'lambda_exp': 0.0,  # a parameter to guarantee the exponential stability o x*
    }

    # Parameters specific to the dynamic system
    dyn_sys_params = {
        'm' : 500.0,   # AH1 mass
        'Jz' : 300.0,  # inertia around z-axis
        'Xu' : 6.106,  # linear drag coefficient - surge
        'Xuu' : 5.0,   # quadratic drag coefficient - surge  'Xuu' : 5.0
        'Nr' : 210.0,  # linear drag coefficient - angular velocity around z-axis
        'Nrr' : 3.0,   # quadratic drag coefficient - angular velocity around z-axis  'Nrr' : 3.0,
        'l1x' : -1.01, 
        'l1y' : -0.353, 
        'alpha1' : np.deg2rad(110.0),
        'l2x' : -1.01,  
        'l2y' : 0.353,
        'alpha2' : np.deg2rad(70.0),
        'l3x' : 0.75,
        'l3y' : 0.0, 
        'alpha3' : np.deg2rad(180.0),
        'h1': 1.,  # nominal health status of actuator 1
        'h2': 1.,  # nominal health status of actuator 2
        'h3': 1.,  # nominal health status of actuator 3
    }

    # faulty cases
    params_faults = {
        'phi_i': [[1, 1, 1], [0, 1, 1], [1, 0, 1], [1, 1, 0]],
    }


    # Postprocessing parameters
    postproc_params = {
        'execute_postprocessing': True,  # True: triggers the generation of the plots described below
        'verbose_info': True,  # print info with high verbosity
        'dpi_': 300,  # DPI number for plots
        'calculate_ROA': True,
        'plot_V': True,
        'plot_Vdot': True,
        'plot_u': True,
        'plot_4D_': False,  # plot 4D Lyapunov f, Lie derivative and control function
        'n_points_4D': 500,
        'n_points_3D': 100,
        'compare_first_last_iters': False,  # saving V, Vdot at the first iter to compare with the final res
        'plot_ctr_weights': False,
        'plot_V_weights': False,
        'plot_dataset': False,
        'save_pdf': False,
        'save_inter_res': False,  
        'saving_points': [0, 5, 50, 500, 2500, 10000, 25000, 49999],

    }

    # Closed-loop system testing parameters
    closed_loop_params = {
        'test_closed_loop_dynamics': False,
        'end_time': 20.0,  # [s] time span of closed-loop tests
        'Dt': 0.01,  # [s] sampling time for Forward Euler integration
        'skip_closed_loop_unconverged': True,  # skip closed-loop tests if the training does not converge
    }

    # joining all the parameters in a single dictionary
    parameters = {**campaign_params,
                  **learner_params,
                  **lyap_params,
                  **control_params,
                  **falsifier_params,
                  **loss_function,
                  **dyn_sys_params,
            	  **params_faults,
                  **postproc_params,
                  **closed_loop_params}


    return parameters, dyn_sys_params

