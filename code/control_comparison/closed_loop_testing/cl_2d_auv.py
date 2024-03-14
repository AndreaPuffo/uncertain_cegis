#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 20 21:04:49 2023

@author: Davide Grande

A function to test the closed-loop dynamics of a 2-dimensional system, given one
ANN controller (passed as 'model').
The function returns the dynamics, the control effort and the Lyapunov value 
V(x) calculated along the trajectory.


"""

import torch 
import matplotlib.pyplot as plt
import numpy as np
import math


def closed_loop_system(samples_number, model, ann_files,
                       des_x1, des_x2,
                       gamma_underbar,
                       control_active_test, 
                       dynamic_fault_test,
                       act1_efficiency_test,
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
                       parameters):

    # parameter to stop the dynamic step when it is diverging
    valid_dynamics = True

    # actuators health status
    h1 = parameters['h1']  
    h2 = parameters['h2']
    h3 = parameters['h3']
    fault_occurred = False

    reference = torch.zeros(1, D_in)
    reference[0, 0] = des_x1  # reference surge speed  [m/s]
    reference[0, 1] = des_x2  # reference sway speed  [m/s]   

    # 1) vanilla ANLC controller
    x_test_hist1 = []
    u_test_hist1 = []
    V_test_hist1 = []

    # Performance measure 1  - RMSE of state error (cumulative)
    diff_s1_c1 = 0.0
    diff_s2_c1 = 0.0

    # Performance measure 2 - RMSE of state error (before and after fault)
    diff_bef_s1_c1 = 0.0
    diff_bef_s2_c1 = 0.0

    diff_aft_s1_c1 = 0.0
    diff_aft_s2_c1 = 0.0

    # Settling times
    sett_thr = 2.0  # settling threshold percentage
    sett_c1 = -1
    thr_reached_c1 = False

    # Performance measure 3  - Total energy consumed
    perfo3_c1 = 0.0


    # Max thrust requested
    max_thrust_c1 = 0.0

    percentage_simul = 1

    for iiter in range(samples_number):
        if valid_dynamics:
            if (iiter == 0):

                # 1) FT-ANLC - controller
                x_0_test1 = torch.zeros(1, D_in)
                err_ref1 = torch.zeros(1, D_in)
                x_0_test1[0, 0] = init_x1
                x_0_test1[0, 1] = init_x2
                x_test1 = x_0_test1  # needed for next iteration step
                err_ref1 = x_0_test1 - reference
                V_test1, outK1 = model.use_in_control_loop(err_ref1)
                outU1 = outK1 * control_active_test
                u_test1 = outU1 * torch.tensor([h1,h2,h3])
                x_test_hist1 = np.append(x_test_hist1, x_0_test1)
                V_test_hist1 = np.append(V_test_hist1, V_test1.detach())
                u_test_hist1 = np.append(u_test_hist1, u_test1.detach())
                # error_hist = np.append(error_hist, err_ref)


                # maximum initial error -- used for performance measurement
                max_err = abs(x_0_test1-reference).max().item()


            else:

                if dynamic_fault_test and not fault_occurred:
                    if (iiter>samples_number/2):
                        if act_faulty_test==1: 
                            h1 = act1_efficiency_test
                        elif act_faulty_test==2:
                            h2 = act1_efficiency_test
                        elif act_faulty_test==3: 
                            h3 = act1_efficiency_test
                        fault_occurred = True
                        print(f"Fault injected on actuator {act_faulty_test} at {iiter*Dt} [s].\n")

                if (iiter / samples_number*100 >= percentage_simul):
                    print(f"Simulation completed: {percentage_simul} %")
                    percentage_simul += 10

                # 1) pFT-ANLC
                V_test1, outK1 = model.use_in_control_loop(err_ref1)
                outU1 = outK1 * control_active_test
                u_test1 = outU1 * torch.tensor([h1,h2,h3])
                x_test_hist1 = np.vstack([x_test_hist1, x_test1.numpy()])
                V_test_hist1 = np.vstack([V_test_hist1, V_test1.detach()])
                u_test_hist1 = np.vstack([u_test_hist1, u_test1.detach()])


            # forward iteration FT-ANLC - controller
            f_next1 = dynamic_sys.dyn(x_test1.detach()[0], u_test1.detach()[0], 
                                      Dt, parameters)  

            x_test1 = torch.zeros(1, D_in)

            for jIn in range(D_in):
                x_test1[0,jIn] = f_next1[0, jIn]

            err_ref1 = x_test1 - reference

            # performance criteria
            diff_s1_c1 += abs(err_ref1[0,0])**2
            diff_s2_c1 += abs(err_ref1[0,1])**2

            if dynamic_fault_test:

                if not fault_occurred:
                    # before fault
                    diff_bef_s1_c1 += abs(err_ref1[0,0])**2
                    diff_bef_s2_c1 += abs(err_ref1[0,1])**2
                else:
                    # after fault
                    diff_aft_s1_c1 += abs(err_ref1[0,0])**2
                    diff_aft_s2_c1 += abs(err_ref1[0,1])**2


            # reaching steady-state target
            if (err_ref1.abs()/max_err).max().item()*100<sett_thr and not thr_reached_c1:
                sett_c1 = end_time/samples_number*iiter
                thr_reached_c1 = True
                print(f"{c1} reached steady-state value at {sett_c1} [s]")

            # leaving steady-state target
            if (err_ref1.abs()/max_err).max().item()*100>sett_thr and thr_reached_c1:
                sett_c1 = end_time/samples_number*iiter
                thr_reached_c1 = False
                print(f"{c1} reached steady-state ESCAPED at {sett_c1} [s]")
                sett_c1 = -1


            # energy consumption
            perfo3_c1 += Dt*u_test1.abs().sum().item()

            # max thrust
            if u_test1.abs().max().item() > max_thrust_c1:
                max_thrust_c1 = u_test1.abs().max().item()

        else:
            break

        # check if any simulation is diverging
        if (x_test1[:,0].abs().max().item() > 20*gamma_overbar):
            valid_dynamics = False # one system is diverging
            print(f"Dynamics diverging at {end_time/samples_number*iiter} [s].")   

    # producing x-axis scale vector
    if valid_dynamics:
        x_axis_scale = np.linspace(0, end_time, samples_number)
    else:
        x_axis_scale = np.linspace(0, end_time/samples_number*iiter, iiter)


    return x_test_hist1, u_test_hist1, V_test_hist1, x_axis_scale, sett_c1


def closed_loop_state_feedback(samples_number, K_lqr1, 
                       des_x1, des_x2,
                       gamma_underbar,
                       control_active_test, 
                       dynamic_fault_test,
                       act1_efficiency_test,
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
                       parameters):

    h1 = parameters['h1']  
    h2 = parameters['h2']
    h3 = parameters['h3']

    # parameter to stop the dynamic step when it is diverging
    valid_dynamics = True

    # actuators health status
    fault_occurred = False

    reference = torch.zeros(1, D_in)
    reference[0, 0] = des_x1  # reference surge speed  [m/s]
    reference[0, 1] = des_x2  # reference sway speed  [m/s]   

    # 1) vanilla ANLC controller
    x_test_hist1 = []
    u_test_hist1 = []

    # Performance measure 1  - RMSE of state error (cumulative)
    diff_s1_c1 = 0.0
    diff_s2_c1 = 0.0

    # Performance measure 2 - RMSE of state error (before and after fault)
    diff_bef_s1_c1 = 0.0
    diff_bef_s2_c1 = 0.0

    diff_aft_s1_c1 = 0.0
    diff_aft_s2_c1 = 0.0

    # Settling times
    sett_thr = 2.0  # settling threshold percentage
    sett_c1 = -1
    thr_reached_c1 = False

    # Performance measure 3  - Total energy consumed
    perfo3_c1 = 0.0


    # Max thrust requested
    max_thrust_c1 = 0.0

    percentage_simul = 1

    for iiter in range(samples_number):
        if valid_dynamics:
            if (iiter == 0):

                # 1) FT-ANLC - controller
                x_0_test1 = torch.zeros(1, D_in)
                err_ref1 = torch.zeros(1, D_in)
                x_0_test1[0, 0] = init_x1
                x_0_test1[0, 1] = init_x2
                x_test1 = x_0_test1  # needed for next iteration step
                err_ref1 = x_0_test1 - reference
                #_, outK1 = model(x_test1)
                outK1 = - torch.mm(torch.from_numpy(K_lqr1), err_ref1.T.double()).T
                outU1 = outK1 * control_active_test
                u_test1 = outU1 * torch.tensor([h1,h2,h3])  
                x_test_hist1 = np.append(x_test_hist1, x_0_test1)
                u_test_hist1 = np.append(u_test_hist1, u_test1.detach())
                # error_hist = np.append(error_hist, err_ref)             


                # maximum initial error -- used for performance measurement
                max_err = abs(x_0_test1-reference).max().item()


            else:

                if dynamic_fault_test and not fault_occurred:
                    if (iiter>samples_number/2):
                        if act_faulty_test==1: 
                            h1 = act1_efficiency_test
                        elif act_faulty_test==2:
                            h2 = act1_efficiency_test
                        elif act_faulty_test==3: 
                            h3 = act1_efficiency_test
                        fault_occurred = True
                        print(f"Fault injected on actuator {act_faulty_test} at {iiter*Dt} [s].\n")

                if (iiter / samples_number*100 >= percentage_simul):
                    print(f"Simulation completed: {percentage_simul} %")
                    percentage_simul += 10

                # 1) pFT-ANLC
                # V_test1, outK1 = model(err_ref1)
                # outU1 = outK1 * control_active_test
                # u_test1 = outU1 * torch.tensor([h1,h2,h3])
                # x_test_hist1 = np.vstack([x_test_hist1, x_test1.numpy()])
                # u_test_hist1 = np.vstack([u_test_hist1, u_test1.detach()])
                
                outK1 = - torch.mm(torch.from_numpy(K_lqr1), err_ref1.T.double()).T
                outU1 = outK1 * control_active_test
                u_test1 = outU1 * torch.tensor([h1,h2,h3]) 
                x_test_hist1 = np.vstack([x_test_hist1, x_test1.numpy()])
                u_test_hist1 = np.vstack([u_test_hist1, u_test1.detach()])
                

            # forward iteration FT-ANLC - controller
            f_next1 = dynamic_sys.dyn(x_test1.detach()[0], u_test1.detach()[0], 
                                      Dt, parameters)  

            x_test1 = torch.zeros(1, D_in)

            for jIn in range(D_in):
                x_test1[0,jIn] = f_next1[0, jIn]

            err_ref1 = x_test1 - reference

            # performance criteria
            diff_s1_c1 += abs(err_ref1[0,0])**2
            diff_s2_c1 += abs(err_ref1[0,1])**2

            if dynamic_fault_test:

                if not fault_occurred:
                    # before fault
                    diff_bef_s1_c1 += abs(err_ref1[0,0])**2
                    diff_bef_s2_c1 += abs(err_ref1[0,1])**2
                else:
                    # after fault
                    diff_aft_s1_c1 += abs(err_ref1[0,0])**2
                    diff_aft_s2_c1 += abs(err_ref1[0,1])**2


            # reaching steady-state target
            if (err_ref1.abs()/max_err).max().item()*100<sett_thr and not thr_reached_c1:
                sett_c1 = end_time/samples_number*iiter
                thr_reached_c1 = True
                print(f"{c1} reached steady-state value at {sett_c1} [s]")

            # leaving steady-state target
            if (err_ref1.abs()/max_err).max().item()*100>sett_thr and thr_reached_c1:
                sett_c1 = end_time/samples_number*iiter
                thr_reached_c1 = False
                print(f"{c1} reached steady-state ESCAPED at {sett_c1} [s]")
                sett_c1 = -1


            # energy consumption
            perfo3_c1 += Dt*u_test1.abs().sum().item()

            # max thrust
            if u_test1.abs().max().item() > max_thrust_c1:
                max_thrust_c1 = u_test1.abs().max().item()

        else:
            break

        # check if any simulation is diverging
        if (x_test1[:,0].abs().max().item() > 20*gamma_overbar):
            valid_dynamics = False # one system is diverging
            print(f"Dynamics diverging at {end_time/samples_number*iiter} [s].")   

    # producing x-axis scale vector
    if valid_dynamics:
        x_axis_scale = np.linspace(0, end_time, samples_number)
    else:
        x_axis_scale = np.linspace(0, end_time/samples_number*iiter, iiter)


    return x_test_hist1, u_test_hist1, x_axis_scale
