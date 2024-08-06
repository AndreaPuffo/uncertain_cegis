#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 14 09:46:12 2024

@authors: Davide Grande
          Andrea Peruffo

 
A script to compare the closed-loop dynamics of several Passive Fault Tolerant controllers.


HOW TO USE ME:
1)The pFT-ANLC controller is loaded from a previous training campaign, saved as: ./results/campaign_'campaign_number').
    If a simulation campaign with several converged controllers is provided, only the first one is used. 

2) Upon providing three state-feedback controllers (e.g. CEGIS-LMI, pFT-ANLC, Hinf), this script
    produces plots for the dynamics and the control efforts.

"""

import os
import torch 
import numpy as np
import closed_loop_testing.cl_2d_auv as cl_single
import systems_dynamics.dynamics_2d_auv_linear as dynamic_sys
from utilities.nn import Net
import postprocessing.plot_2d_auv as plot_controllers
import configuration.config_2d_auv_faulty_linear as config_file


'''
Parameters
'''
campaign_number = 500021  # number of the campaign number to load
compare_with_StateFeedback = True  # if True: compare pFT-ANLC vs "K_Hinf1 and K_Hinf2"
compare_three_single = True   # if True: compare pFT-ANLC [0] vs "K_single" vs "K_Hinf1" --> requires compare_with_StateFeedback = True

end_time = 30.
Dt = 0.01

parameters, dyn_sys_params = config_file.set_params()
x_star = parameters['x_star']
gamma_underbar = parameters['gamma_underbar']  # defined as 'rho overbar' in the paper
gamma_overbar = parameters['gamma_overbar']   # defined as 'rho underbar' in the paper
dpi_ = parameters['dpi_']  # DPI number for plots
D_in = parameters['n_input'] 
plot_saturation = False
saturation_value = 37.1


c1, c2, c3,  = "pFT-ANLC", "pFT-ANLC", "pFT-ANLC"
c6, c7 =  "$H_\infty$ - conservative", "CEGIS-based"
plot_c6 = False
plot_c7 = True

#--------------------------------campaign_400084
# conservative 40% error -- with limit r/u = 880
K_Hinf1= np.array([[232.1081,  277.2772],
                    [183.4074, -219.4158],
                    [-0.0298, -776.4082]])


# conservative 50% error -- with limit r/u = 50
K_Hinf2= 1.0e+03*np.array([[0.1276,    0.1518],
                            [0.1422,   -0.1698],
                            [0.0011,   -1.3107]])

# Controller CEGIS_LMI
# no saturation
K_single = -np.array([[-296.9619804,    65.18469131],
                     [-206.61813474, -109.80165865],
                     [ -96.94795656,  -31.35621447]])

# no saturation -- medium conservative
K_single = -np.array([[-771.21922226,    1.29314791],
                    [-913.86388724,   30.75392823],
                    [  -2.87588309,  117.7393848 ]])

## This one works - CEGIS-LMI aggressive. Steady-state: 2.5s
K_single = -np.array([[-5.06253082e+02,  6.30357945e+01],
                    [-5.06253082e+02, -6.30357945e+01],
                    [ 1.47713717e-09,  4.19876431e+02]])


# # old doing funny things -- here the CEGIS converges very slowly on the x2 dynamics
# K_single = -np.array([[-3.73232728e+01,  7.34473330e+01],
#                      [-3.73232728e+01, -7.34473330e+01],
#                      [-6.77357684e-11, -9.18628044e+01]])

# ## This one works - CEGIS-LMI conservative. Steady-state: 45s
# K_single = -np.array([[-4.30634256,   22.98265196],
#                     [-9.12111491,  -73.6798391],
#                     [-24.60223679, -124.74370083]])


#--------------------------------



# Loading ANN
folder_results = "results/campaign_" + str(campaign_number)
current_dir = os.getcwd()
final_dir = current_dir + "/" + folder_results + "/converged_controllers/"

ann_files = 0
for path in os.listdir(final_dir):
    if os.path.isfile(os.path.join(final_dir, path)):
        ann_files += 1

ann_list = []
# list to store files
for path in os.listdir(final_dir):
    # check if current path is a file
    if os.path.isfile(os.path.join(final_dir, path)):
        name_file = path
        print(name_file)

    path_ann = final_dir + "/" + name_file
    seed_ = 0  #anything random here

    # generated random (linear) control vector if not initialised
    if not parameters['control_initialised']:
        parameters['init_control'] = torch.rand([parameters['size_ctrl_layers'][-1], parameters['n_input']]) - 0.5

    model = Net(parameters, seed_)
    model.load_state_dict(torch.load(path_ann))
    model.eval()

    ann_list.append(model)


# 0) Generating result folder
try:
    res_dir = final_dir + "results_aposteriori/"
    os.mkdir(res_dir)
except OSError:
    print("Creation of the result directory %s failed" % res_dir)
else:
    print("Result directory successfully created as: \n %s \n" % res_dir)


'''
Testing
'''
print("Run closed-loop test.")    
samples_number = int(end_time / Dt)
contr_out = model.size_ctrl_layers[-1]


# testing velocity following
initial_x1 =     [  -gamma_overbar,       -gamma_overbar/2,     -gamma_overbar/2,    -gamma_overbar/2] # [m/s]
initial_x2 =     [  -gamma_overbar,        gamma_overbar/4,      gamma_overbar/4,     gamma_overbar/4] # [rad/s]
desired_x1 =     [x_star[0].item(),        x_star[0].item(),    x_star[0].item(),    x_star[0].item()] # [m/s]
desired_x2 =     [x_star[1].item(),        x_star[1].item(),    x_star[1].item(),    x_star[1].item()] # [rad/s]
dynamic_fault =  [0,                                      1,                   1,                   1] #
act_efficiency = [1,                                    0.0,                 0.0,                 0.0] #
act_faulty =     [-1,                                     1,                   2,                   3] #
control_active = [1,                                      1,                   1,                   1] #


for iTest in range(len(initial_x1)):
    message = "\nClosed-loop " + str(iTest + 1) + "/" + str(len(initial_x1))
    print(message)

    final_dir_ = res_dir + "test_#" + str(iTest) + "/"
    os.mkdir(final_dir_)

    # re-setting actualtor health stati
    parameters['h1']=1.0
    parameters['h2']=1.0
    parameters['h3']=1.0

    des_x1 = desired_x1[iTest]
    des_x2 = desired_x2[iTest]
    init_x1 = initial_x1[iTest]
    init_x2 = initial_x2[iTest]

    control_active_test = control_active[iTest]
    dynamic_fault_test = dynamic_fault[iTest]
    act_efficiency_test = act_efficiency[iTest]
    act_faulty_test = act_faulty[iTest]
    if act_faulty==1:
        parameters['h1']=0.0
    elif act_faulty==2:
        parameters['h2']=0.0
    elif act_faulty==3:
        parameters['h3']=0.0
    
    
    x_all = [None] * ann_files
    u_all = [None] * ann_files
    V_all = [None] * ann_files

    # Simulating pFT-ANLC
    for jContr in range(ann_files):
        model = ann_list[jContr] # TODO

        print(f"\nTest {iTest + 1}/{len(initial_x1)} - model #{jContr+1}/{ann_files}")

        # closed-loop: pFT-ANLC controllers
        x_test_hist1, u_test_hist1, V_test_hist1, x_axis_scale, sett_c1 = \
            cl_single.closed_loop_system(
                                         samples_number, model, ann_files,
                                         des_x1, des_x2,
                                         gamma_underbar,
                                         control_active_test, 
                                         dynamic_fault_test,
                                         act_efficiency_test,
                                         act_faulty_test,
                                         init_x1, init_x2,
                                         Dt, end_time, gamma_overbar,
                                         D_in,
                                         final_dir_,
                                         dyn_sys_params, dpi_,
                                         dynamic_sys,
                                         c1,
                                         x_star, 
                                         contr_out,
                                         parameters) 

        x_all[jContr] = x_test_hist1.copy()
        u_all[jContr] = u_test_hist1.copy()
        V_all[jContr] = V_test_hist1.copy()
    
    if compare_with_StateFeedback:
        ########################################################################
        # Simulating State Feedback (Hinf1 and Hinf2)
        x_all_state_f = [None] * 2
        u_all_state_f = [None] * 2

        # single State-Feedback tuning
        x_single = [None]
        u_single = [None]

    
        print(f"\nTest {iTest + 1}/{len(initial_x1)} - State-feedback #1/2")
    
        # closed-loop: StateFeedback #1
        x_test_hist1, u_test_hist1, x_axis_scale = \
            cl_single.closed_loop_state_feedback(
                                         samples_number, K_Hinf1, 
                                         des_x1, des_x2,
                                         gamma_underbar,
                                         control_active_test, 
                                         dynamic_fault_test,
                                         act_efficiency_test,
                                         act_faulty_test,
                                         init_x1, init_x2,
                                         Dt, end_time, gamma_overbar,
                                         D_in,
                                         final_dir_,
                                         dyn_sys_params, dpi_,
                                         dynamic_sys,
                                         c6,
                                         x_star, 
                                         contr_out,
                                         parameters) 
    
        x_all_state_f[0] = x_test_hist1.copy()
        u_all_state_f[0] = u_test_hist1.copy()
           
        print(f"\nTest {iTest + 1}/{len(initial_x1)} - State-feedback #2/2")

        # closed-loop: StateFeedback #2
        x_test_hist1, u_test_hist1, x_axis_scale = \
            cl_single.closed_loop_state_feedback(
                                         samples_number, K_Hinf2, 
                                         des_x1, des_x2,
                                         gamma_underbar,
                                         control_active_test, 
                                         dynamic_fault_test,
                                         act_efficiency_test,
                                         act_faulty_test,
                                         init_x1, init_x2,
                                         Dt, end_time, gamma_overbar,
                                         D_in,
                                         final_dir_,
                                         dyn_sys_params, dpi_,
                                         dynamic_sys,
                                         c6,
                                         x_star, 
                                         contr_out,
                                         parameters) 
    
        x_all_state_f[1] = x_test_hist1.copy()
        u_all_state_f[1] = u_test_hist1.copy()
    
      
        # Comparison and plotting

        # Single state-feedback law
        print(f"\nTest {iTest + 1}/{len(initial_x1)} - State-feedback single tuning #1/1")

        # closed-loop: StateFeedback #2
        x_test_hist1, u_test_hist1, x_axis_scale  = \
            cl_single.closed_loop_state_feedback(
                                        samples_number, K_single, 
                                        des_x1, des_x2,
                                        gamma_underbar,
                                        control_active_test, 
                                        dynamic_fault_test,
                                        act_efficiency_test,
                                        act_faulty_test,
                                        init_x1, init_x2,
                                        Dt, end_time, gamma_overbar,
                                        D_in,
                                        final_dir_,
                                        dyn_sys_params, dpi_,
                                        dynamic_sys,
                                        c7,
                                        x_star, 
                                        contr_out,
                                        parameters) 

        x_single[0] = x_test_hist1.copy()
        u_single[0] = u_test_hist1.copy()


        # plot all "pFT-ANLC [0]" vs "K_Hinf1" vs "K_single"
        plot_controllers.plot_comparison_three_single(x_all, u_all, V_all, x_axis_scale,
                                                    dynamic_fault_test, c1, c6, 
                                                    x_single, u_single, c7,
                                                    plot_c6, plot_c7,
                                                    act_faulty_test,
                                                    des_x1, des_x2, init_x1, init_x2, gamma_underbar,
                                                    plot_saturation,
                                                    saturation_value,
                                                    samples_number, ann_files,
                                                    x_all_state_f, u_all_state_f,
                                                    dpi_, final_dir_)

