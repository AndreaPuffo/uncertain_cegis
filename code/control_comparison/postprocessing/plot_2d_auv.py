#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 20 22:05:22 2023

@author: Davide Grande

A script to plot the results of the tests.

"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter


def plot(x_all, u_all, V_all, x_axis_scale, 
         dynamic_fault_test, c1, act_faulty_test,
         des_x1, des_x2, init_x1, init_x2, 
         gamma_underbar, gamma_overbar,
         samples_number, ann_files, 
         dpi_, final_dir_):
    # This function plots the results of a simulation campaign in terms of 
    # dynamics and control effort bounds.
    
    # Obtaining boundaries
    min_x1 = x_all[0][:,0].copy()
    min_x2 = x_all[0][:,1].copy()
    min_u1 = u_all[0][:,0].copy()
    min_u2 = u_all[0][:,1].copy()
    min_u3 = u_all[0][:,2].copy()
    min_V = V_all[0][:,0].copy()
   
    max_x1 = x_all[0][:,0].copy()
    max_x2 = x_all[0][:,1].copy()
    max_u1 = u_all[0][:,0].copy()
    max_u2 = u_all[0][:,1].copy()
    max_u3 = u_all[0][:,2].copy()
    max_V = V_all[0][:,0].copy()


    for jCntr in range(ann_files):
        min_x1 = np.minimum(min_x1, x_all[jCntr][:,0]).copy()
        min_x2 = np.minimum(min_x2, x_all[jCntr][:,1]).copy()
        min_u1 = np.minimum(min_u1, u_all[jCntr][:,0]).copy()
        min_u2 = np.minimum(min_u2, u_all[jCntr][:,1]).copy()
        min_u3 = np.minimum(min_u3, u_all[jCntr][:,2]).copy()
        min_V = np.minimum(min_V, V_all[jCntr][:,0]).copy()

        max_x1 = np.maximum(max_x1, x_all[jCntr][:,0]).copy()
        max_x2 = np.maximum(max_x2, x_all[jCntr][:,1]).copy()
        max_u1 = np.maximum(max_u1, u_all[jCntr][:,0]).copy()
        max_u2 = np.maximum(max_u2, u_all[jCntr][:,1]).copy()
        max_u3 = np.maximum(max_u3, u_all[jCntr][:,2]).copy()
        max_V = np.maximum(max_V, V_all[jCntr][:,0]).copy()
    
    
    # 1.1 - all forces together
    title_fig_c = "Control_input_forces_bounds.png"
    fig = plt.figure()
    plt.fill_between(x_axis_scale, min_u1, max_u1, facecolor='blue', alpha=0.8, interpolate=True, label='$F_1$')
    plt.fill_between(x_axis_scale, min_u2, max_u2, facecolor='red', alpha=0.8, interpolate=True, label='$F_2$')
    plt.fill_between(x_axis_scale, min_u3, max_u3, facecolor='green', alpha=0.8, interpolate=True, label='$F_3$')
    plt.legend(loc='best')
    plt.xlabel("Time [s]")
    plt.ylabel("Force [N]")
    plt.grid()
    plt.savefig(final_dir_ + title_fig_c, dpi=dpi_)
    plt.close(fig)
    
            
    
    # 1.1 - all forces together - half time 
    t_short = 2
    title_fig_c = "Control_input_forces_bounds_t" + str(t_short) + ".png"
    fig = plt.figure()
    plt.fill_between(x_axis_scale[:int(samples_number/t_short)], min_u1[:int(samples_number/t_short)], max_u1[:int(samples_number/t_short)], facecolor='blue', alpha=0.8, interpolate=True, label='$F_1$')
    plt.fill_between(x_axis_scale[:int(samples_number/t_short)], min_u2[:int(samples_number/t_short)], max_u2[:int(samples_number/t_short)], facecolor='red', alpha=0.8, interpolate=True, label='$F_2$')
    plt.fill_between(x_axis_scale[:int(samples_number/t_short)], min_u3[:int(samples_number/t_short)], max_u3[:int(samples_number/t_short)], facecolor='green', alpha=0.8, interpolate=True, label='$F_3$')
    plt.legend(loc='best')
    plt.xlabel("Time [s]")
    plt.ylabel("Force [N]")
    plt.grid()
    plt.savefig(final_dir_ + title_fig_c, dpi=dpi_)
    plt.close(fig)
    

    t_short = 6
    title_fig_c = "Control_input_forces_bounds_t" + str(t_short) + ".png"
    fig = plt.figure()
    plt.fill_between(x_axis_scale[:int(samples_number/t_short)], min_u1[:int(samples_number/t_short)], max_u1[:int(samples_number/t_short)], facecolor='blue', alpha=0.8, interpolate=True, label='$F_1$')
    plt.fill_between(x_axis_scale[:int(samples_number/t_short)], min_u2[:int(samples_number/t_short)], max_u2[:int(samples_number/t_short)], facecolor='red', alpha=0.8, interpolate=True, label='$F_2$')
    plt.fill_between(x_axis_scale[:int(samples_number/t_short)], min_u3[:int(samples_number/t_short)], max_u3[:int(samples_number/t_short)], facecolor='green', alpha=0.8, interpolate=True, label='$F_3$')
    plt.legend(loc='best')
    plt.xlabel("Time [s]")
    plt.ylabel("Force [N]")
    plt.grid()
    plt.savefig(final_dir_ + title_fig_c, dpi=dpi_)
    plt.close(fig)


    if dynamic_fault_test:
        # 1.1 - all forces together (final section)
        title_fig_c = "Control_input_forces_short_bounds.png"
        fig = plt.figure()
        plt.fill_between(x_axis_scale[int(samples_number/2-0.1*samples_number):], min_u1[int(samples_number/2-0.1*samples_number):], max_u1[int(samples_number/2-0.1*samples_number):], facecolor='blue', alpha=0.8, interpolate=True, label='$F_1$')
        plt.fill_between(x_axis_scale[int(samples_number/2-0.1*samples_number):], min_u2[int(samples_number/2-0.1*samples_number):], max_u2[int(samples_number/2-0.1*samples_number):], facecolor='red', alpha=0.8, interpolate=True, label='$F_2$')
        plt.fill_between(x_axis_scale[int(samples_number/2-0.1*samples_number):], min_u3[int(samples_number/2-0.1*samples_number):], max_u3[int(samples_number/2-0.1*samples_number):], facecolor='green', alpha=0.8, interpolate=True, label='$F_3$')
        plt.legend(loc='best')
        plt.xlabel("Time [s]")
        plt.ylabel("Force [N]")
        plt.grid()
        plt.savefig(final_dir_ + title_fig_c, dpi=dpi_)
        plt.close(fig)
        

        title_fig_c = "Control_input_forces_bounds_v2.png"
        fig = plt.figure()
        
        plt.fill_between(x_axis_scale, min_u1, max_u1, facecolor='blue', alpha=0.8, interpolate=True, label='$F_1$')
        plt.fill_between(x_axis_scale, min_u2, max_u2, facecolor='red', alpha=0.8, interpolate=True, label='$F_2$')
        plt.fill_between(x_axis_scale, min_u3, max_u3, facecolor='green', alpha=0.8, interpolate=True, label='$F_3$')
        
        if act_faulty_test==1:
            fault_location_y = (max_u1[int(samples_number/2)] - min_u1[int(samples_number/2)])/2 + min_u1[int(samples_number/2)]
        if act_faulty_test==2:
            fault_location_y = (max_u2[int(samples_number/2)] - min_u2[int(samples_number/2)])/2 + min_u2[int(samples_number/2)]
        if act_faulty_test==3:
            fault_location_y = (max_u3[int(samples_number/2)] - min_u3[int(samples_number/2)])/2 + min_u3[int(samples_number/2)]
        
        plt.scatter(x_axis_scale[int(samples_number/2)], fault_location_y, s=100, c='red', label='$F_{%s}$ - fault injected' %act_faulty_test)
        plt.legend(loc='best')
        plt.xlabel("Time [s]")
        plt.ylabel("Force [N]")
        plt.grid()
        plt.savefig(final_dir_ + title_fig_c, dpi=dpi_)
        plt.close(fig)
        
        
        
        perc = 0.2  # percentage of time plot before and after fault   
        title_fig_c = "Control_input_forces_bounds_" + str(perc) + "perc.png"
        t_from = int(samples_number/2-perc*samples_number)
        t_to= int(samples_number/2+perc*samples_number)
        fig = plt.figure()
        plt.fill_between(x_axis_scale[t_from:t_to], min_u1[t_from:t_to], max_u1[t_from:t_to], facecolor='blue', alpha=0.8, interpolate=True, label='$F_1$')
        plt.fill_between(x_axis_scale[t_from:t_to], min_u2[t_from:t_to], max_u2[t_from:t_to], facecolor='red', alpha=0.8, interpolate=True, label='$F_2$')
        plt.fill_between(x_axis_scale[t_from:t_to], min_u3[t_from:t_to], max_u3[t_from:t_to], facecolor='green', alpha=0.8, interpolate=True, label='$F_3$')
        if act_faulty_test==1:
            fault_location_y = (max_u1[int(samples_number/2)] - min_u1[int(samples_number/2)])/2 + min_u1[int(samples_number/2)]
        if act_faulty_test==2:
            fault_location_y = (max_u2[int(samples_number/2)] - min_u2[int(samples_number/2)])/2 + min_u2[int(samples_number/2)]
        if act_faulty_test==3:
            fault_location_y = (max_u3[int(samples_number/2)] - min_u3[int(samples_number/2)])/2 + min_u3[int(samples_number/2)]
        plt.scatter(x_axis_scale[int(samples_number/2)], fault_location_y, s=100, c='red', label='$F_{%s}$ - fault injected' %act_faulty_test)
        plt.legend(loc='best')
        plt.xlabel("Time [s]")
        plt.ylabel("Force [N]")
        plt.grid()
        plt.savefig(final_dir_ + title_fig_c, dpi=dpi_)
        plt.close(fig)
        
        
        
        perc = 0.1  # percentage of time plot before and after fault   
        title_fig_c = "Control_input_forces_bounds_" + str(perc) + "perc.png"
        t_from = int(samples_number/2-perc*samples_number)
        t_to= int(samples_number/2+perc*samples_number)
        fig = plt.figure()
        plt.fill_between(x_axis_scale[t_from:t_to], min_u1[t_from:t_to], max_u1[t_from:t_to], facecolor='blue', alpha=0.8, interpolate=True, label='$F_1$')
        plt.fill_between(x_axis_scale[t_from:t_to], min_u2[t_from:t_to], max_u2[t_from:t_to], facecolor='red', alpha=0.8, interpolate=True, label='$F_2$')
        plt.fill_between(x_axis_scale[t_from:t_to], min_u3[t_from:t_to], max_u3[t_from:t_to], facecolor='green', alpha=0.8, interpolate=True, label='$F_3$')
        if act_faulty_test==1:
            fault_location_y = (max_u1[int(samples_number/2)] - min_u1[int(samples_number/2)])/2 + min_u1[int(samples_number/2)]
        if act_faulty_test==2:
            fault_location_y = (max_u2[int(samples_number/2)] - min_u2[int(samples_number/2)])/2 + min_u2[int(samples_number/2)]
        if act_faulty_test==3:
            fault_location_y = (max_u3[int(samples_number/2)] - min_u3[int(samples_number/2)])/2 + min_u3[int(samples_number/2)]
        plt.scatter(x_axis_scale[int(samples_number/2)], fault_location_y, s=100, c='red', label='$F_{%s}$ - fault injected' %act_faulty_test)
        plt.legend(loc='best')
        plt.xlabel("Time [s]")
        plt.ylabel("Force [N]")
        plt.grid()
        plt.savefig(final_dir_ + title_fig_c, dpi=dpi_)
        plt.close(fig)
        

        
    title_fig_ref = "Reference_dynamics_x1_bounds.png"
    fig = plt.figure()
    plt.plot(x_axis_scale, min_x1*0 + des_x1, '--g', label='$x_1^\star$')  # reference
    plt.fill_between(x_axis_scale, min_x1, 
                     max_x1, facecolor='blue', alpha=0.5, interpolate=True, label=c1)
    plt.fill_between(x_axis_scale, min_x1*0 + des_x1+gamma_underbar, 
                     min_x1*0 + des_x1-gamma_underbar, facecolor='green', alpha=0.2, interpolate=True, label='$\epsilon$-stability bound')
    plt.xlabel("Time [s]")
    plt.ylabel("Surge speed [m/s]")
    plt.legend(loc='best')
    plt.grid()
    plt.savefig(final_dir_ + title_fig_ref, dpi=dpi_)
    plt.close(fig)



    t_short = 2
    title_fig_ref = "Reference_dynamics_x1_bounds_t" + str(t_short) + ".png"
    fig = plt.figure()
    plt.plot(x_axis_scale[:int(samples_number/t_short)], min_x1[:int(samples_number/t_short)]*0 + des_x1, '--g', label='$x_1^\star$')  # reference
    plt.fill_between(x_axis_scale[:int(samples_number/t_short)], min_x1[:int(samples_number/t_short)], 
                     max_x1[:int(samples_number/t_short)], facecolor='blue', alpha=0.5, interpolate=True, label=c1)
    plt.fill_between(x_axis_scale[:int(samples_number/t_short)], min_x1[:int(samples_number/t_short)]*0 + des_x1+gamma_underbar, 
                     min_x1[:int(samples_number/t_short)]*0 + des_x1-gamma_underbar, facecolor='green', alpha=0.2, interpolate=True, label='$\epsilon$-stability bound')
    plt.xlabel("Time [s]")
    plt.ylabel("Surge speed [m/s]")
    plt.legend(loc='lower right')
    plt.grid()
    plt.savefig(final_dir_ + title_fig_ref, dpi=dpi_)
    plt.close(fig)


    t_short = 6
    title_fig_ref = "Reference_dynamics_x1_bounds_t" + str(t_short) + ".png"
    fig = plt.figure()
    plt.plot(x_axis_scale[:int(samples_number/t_short)], min_x1[:int(samples_number/t_short)]*0 + des_x1, '--g', label='$x_1^\star$')  # reference
    plt.fill_between(x_axis_scale[:int(samples_number/t_short)], min_x1[:int(samples_number/t_short)], 
                     max_x1[:int(samples_number/t_short)], facecolor='blue', alpha=0.5, interpolate=True, label=c1)
    plt.fill_between(x_axis_scale[:int(samples_number/t_short)], min_x1[:int(samples_number/t_short)]*0 + des_x1+gamma_underbar, 
                     min_x1[:int(samples_number/t_short)]*0 + des_x1-gamma_underbar, facecolor='green', alpha=0.2, interpolate=True, label='$\epsilon$-stability bound')
    plt.xlabel("Time [s]")
    plt.ylabel("Surge speed [m/s]")
    plt.legend(loc='lower right')
    plt.grid()
    plt.savefig(final_dir_ + title_fig_ref, dpi=dpi_)
    plt.close(fig)


    if dynamic_fault_test:
        
        title_fig_ref = "Reference_dynamics_x1_bounds_short.png"
        fig = plt.figure()
        plt.plot(x_axis_scale[int(samples_number/2-0.1*samples_number):], min_x1[int(samples_number/2-0.1*samples_number):]*0 + des_x1, '--g', label='$x_1^\star$')  # reference
        plt.fill_between(x_axis_scale[int(samples_number/2-0.1*samples_number):], min_x1[int(samples_number/2-0.1*samples_number):], 
                         max_x1[int(samples_number/2-0.1*samples_number):], facecolor='blue', alpha=0.5, interpolate=True, label=c1)
        plt.fill_between(x_axis_scale[int(samples_number/2-0.1*samples_number):], min_x1[int(samples_number/2-0.1*samples_number):]*0 + des_x1+gamma_underbar, 
                         min_x1[int(samples_number/2-0.1*samples_number):]*0 + des_x1-gamma_underbar, facecolor='green', alpha=0.2, interpolate=True, label='$\epsilon$-stability bound')
        plt.xlabel("Time [s]")
        plt.ylabel("Surge speed [m/s]")
        plt.legend(loc='best')
        plt.grid()
        plt.savefig(final_dir_ + title_fig_ref, dpi=dpi_)
        plt.close(fig)


        fault_location_y = (max_x1[int(samples_number/2)] - min_x1[int(samples_number/2)])/2 + min_x1[int(samples_number/2)]
        title_fig_ref = "Reference_dynamics_x1_bounds_short_v2.png"
        fig = plt.figure()
        plt.plot(x_axis_scale[int(samples_number/2-0.1*samples_number):], min_x1[int(samples_number/2-0.1*samples_number):]*0 + des_x1, '--g', label='$x_1^\star$')  # reference
        plt.fill_between(x_axis_scale[int(samples_number/2-0.1*samples_number):], min_x1[int(samples_number/2-0.1*samples_number):], 
                         max_x1[int(samples_number/2-0.1*samples_number):], facecolor='blue', alpha=0.5, interpolate=True, label=c1)
        plt.fill_between(x_axis_scale[int(samples_number/2-0.1*samples_number):], min_x1[int(samples_number/2-0.1*samples_number):]*0 + des_x1+gamma_underbar, 
                         min_x1[int(samples_number/2-0.1*samples_number):]*0 + des_x1-gamma_underbar, facecolor='green', alpha=0.2, interpolate=True, label='$\epsilon$-stability bound')
        plt.scatter(x_axis_scale[int(samples_number/2)], fault_location_y, s=200, c='red', label='$F_{%s}$ - fault injected' %act_faulty_test)
        plt.xlabel("Time [s]")
        plt.ylabel("Surge speed [m/s]")
        plt.legend(loc='best')
        plt.grid()
        plt.savefig(final_dir_ + title_fig_ref, dpi=dpi_)
        plt.close(fig)
        
        
        perc = 0.1  # percentage of time plot before and after fault   
        title_fig_c = "Reference_dynamics_x1_bounds_" + str(perc) + "perc.png"
        t_from = int(samples_number/2-perc*samples_number)
        t_to= int(samples_number/2+perc*samples_number)
        fig = plt.figure()
        plt.plot(x_axis_scale[t_from:t_to], min_x1[t_from:t_to]*0 + des_x1, '--g', label='$x_1^\star$')  # reference
        plt.fill_between(x_axis_scale[t_from:t_to], min_x1[t_from:t_to], 
                         max_x1[t_from:t_to], facecolor='blue', alpha=0.5, interpolate=True, label=c1)
        plt.fill_between(x_axis_scale[t_from:t_to], min_x1[t_from:t_to]*0 + des_x1+gamma_underbar, 
                         min_x1[t_from:t_to]*0 + des_x1-gamma_underbar, facecolor='green', alpha=0.2, interpolate=True, label='$\epsilon$-stability bound')
        if act_faulty_test==1:
            fault_location_y = (max_u1[int(samples_number/2)] - min_u1[int(samples_number/2)])/2 + min_u1[int(samples_number/2)]
        if act_faulty_test==2:
            fault_location_y = (max_u2[int(samples_number/2)] - min_u2[int(samples_number/2)])/2 + min_u2[int(samples_number/2)]
        if act_faulty_test==3:
            fault_location_y = (max_u3[int(samples_number/2)] - min_u3[int(samples_number/2)])/2 + min_u3[int(samples_number/2)]
        plt.scatter(x_axis_scale[int(samples_number/2)], fault_location_y, s=100, c='red', label='$F_{%s}$ - fault injected' %act_faulty_test)
        plt.legend(loc='best')
        plt.xlabel("Time [s]")
        plt.ylabel("Force [N]")
        plt.grid()
        plt.savefig(final_dir_ + title_fig_c, dpi=dpi_)
        plt.close(fig)
        
        
        perc = 0.2  # percentage of time plot before and after fault   
        title_fig_c = "Reference_dynamics_x1_bounds_" + str(perc) + "perc.png"
        t_from = int(samples_number/2-perc*samples_number)
        t_to= int(samples_number/2+perc*samples_number)
        fig = plt.figure()
        plt.plot(x_axis_scale[t_from:t_to], min_x1[t_from:t_to]*0 + des_x1, '--g', label='$x_1^\star$')  # reference
        plt.fill_between(x_axis_scale[t_from:t_to], min_x1[t_from:t_to], 
                         max_x1[t_from:t_to], facecolor='blue', alpha=0.5, interpolate=True, label=c1)
        plt.fill_between(x_axis_scale[t_from:t_to], min_x1[t_from:t_to]*0 + des_x1+gamma_underbar, 
                         min_x1[t_from:t_to]*0 + des_x1-gamma_underbar, facecolor='green', alpha=0.2, interpolate=True, label='$\epsilon$-stability bound')
        fault_location_y = (max_x1[int(samples_number/2)] - min_x1[int(samples_number/2)])/2 + min_x1[int(samples_number/2)]
        plt.scatter(x_axis_scale[int(samples_number/2)], fault_location_y, s=100, c='red', label='$F_{%s}$ - fault injected' %act_faulty_test)
        plt.legend(loc='best')
        plt.xlabel("Time [s]")
        plt.ylabel("Force [N]")
        plt.grid()
        plt.savefig(final_dir_ + title_fig_c, dpi=dpi_)
        plt.close(fig)
        


    # 4) Dynamics x2
    title_fig_ref = "Reference_dynamics_x2_bounds.png"
    fig = plt.figure()
    plt.plot(x_axis_scale, min_x2*0 + des_x2, '--g', label='$x_2^\star$')  # reference
    plt.fill_between(x_axis_scale, min_x2, 
                     max_x2, facecolor='blue', alpha=0.5, interpolate=True, label=c1)
    plt.fill_between(x_axis_scale, min_x2*0 + des_x2+gamma_underbar, 
                     min_x2*0 + des_x2-gamma_underbar, facecolor='green', alpha=0.2, interpolate=True, label='$\epsilon$-stability bound')
    plt.xlabel("Time [s]")
    plt.ylabel("Angular speed [rad/s]")
    plt.legend(loc='lower right')
    plt.grid()
    plt.savefig(final_dir_ + title_fig_ref, dpi=dpi_)
    plt.close(fig)





    # 4) Dynamics x2
    t_short = 2
    title_fig_ref = "Reference_dynamics_x2_bounds_t" + str(t_short) + ".png"
    fig = plt.figure()
    plt.plot(x_axis_scale[:int(samples_number/t_short)], min_x2[:int(samples_number/t_short)]*0 + des_x2, '--g', label='$x_2^\star$')  # reference
    plt.fill_between(x_axis_scale[:int(samples_number/t_short)], min_x2[:int(samples_number/t_short)], 
                     max_x2[:int(samples_number/t_short)], facecolor='blue', alpha=0.5, interpolate=True, label=c1)
    plt.fill_between(x_axis_scale[:int(samples_number/t_short)], min_x2[:int(samples_number/t_short)]*0 + des_x2+gamma_underbar, 
                     min_x2[:int(samples_number/t_short)]*0 + des_x2-gamma_underbar, facecolor='green', alpha=0.2, interpolate=True, label='$\epsilon$-stability bound')
    plt.xlabel("Time [s]")
    plt.ylabel("Angular speed [rad/s]")
    plt.legend(loc='lower right')
    plt.grid()
    plt.savefig(final_dir_ + title_fig_ref, dpi=dpi_)
    plt.close(fig)


    # 4) Dynamics x2
    t_short = 6
    title_fig_ref = "Reference_dynamics_x2_bounds_t" + str(t_short) + ".png"
    fig = plt.figure()
    plt.plot(x_axis_scale[:int(samples_number/t_short)], min_x2[:int(samples_number/t_short)]*0 + des_x2, '--g', label='$x_2^\star$')  # reference
    plt.fill_between(x_axis_scale[:int(samples_number/t_short)], min_x2[:int(samples_number/t_short)], 
                     max_x2[:int(samples_number/t_short)], facecolor='blue', alpha=0.5, interpolate=True, label=c1)
    plt.fill_between(x_axis_scale[:int(samples_number/t_short)], min_x2[:int(samples_number/t_short)]*0 + des_x2+gamma_underbar, 
                     min_x2[:int(samples_number/t_short)]*0 + des_x2-gamma_underbar, facecolor='green', alpha=0.2, interpolate=True, label='$\epsilon$-stability bound')
    plt.xlabel("Time [s]")
    plt.ylabel("Angular speed [rad/s]")
    plt.legend(loc='lower right')
    plt.grid()
    plt.savefig(final_dir_ + title_fig_ref, dpi=dpi_)
    plt.close(fig)


    if dynamic_fault_test:
        
        title_fig_ref = "Reference_dynamics_x2_short_bounds.png"
        fig = plt.figure()
        plt.plot(x_axis_scale[int(samples_number/2-0.1*samples_number):], min_x2[int(samples_number/2-0.1*samples_number):]*0 + des_x2, '--g', label='$x_2^\star$')  # reference
        plt.fill_between(x_axis_scale[int(samples_number/2-0.1*samples_number):], min_x2[int(samples_number/2-0.1*samples_number):], 
                         max_x2[int(samples_number/2-0.1*samples_number):], facecolor='blue', alpha=0.5, interpolate=True, label=c1)
        plt.fill_between(x_axis_scale[int(samples_number/2-0.1*samples_number):], min_x2[int(samples_number/2-0.1*samples_number):]*0 + des_x2+gamma_underbar, 
                         min_x2[int(samples_number/2-0.1*samples_number):]*0 + des_x2-gamma_underbar, facecolor='green', alpha=0.2, interpolate=True, label='$\epsilon$-stability bound')
        plt.xlabel("Time [s]")
        plt.ylabel("Angular speed [rad/s]")
        plt.legend(loc='best')
        plt.grid()
        plt.savefig(final_dir_ + title_fig_ref, dpi=dpi_)
        plt.close(fig)


        fault_location_y = (max_x2[int(samples_number/2)] - min_x2[int(samples_number/2)])/2 + min_x2[int(samples_number/2)]
        title_fig_ref = "Reference_dynamics_x2_short_v2_bounds.png"
        fig = plt.figure()
        plt.plot(x_axis_scale[int(samples_number/2-0.1*samples_number):], min_x2[int(samples_number/2-0.1*samples_number):]*0 + des_x2, '--g', label='$x_2^\star$')  # reference
        plt.fill_between(x_axis_scale[int(samples_number/2-0.1*samples_number):], min_x2[int(samples_number/2-0.1*samples_number):], 
                         max_x2[int(samples_number/2-0.1*samples_number):], facecolor='blue', alpha=0.5, interpolate=True, label=c1)
        plt.fill_between(x_axis_scale[int(samples_number/2-0.1*samples_number):], min_x2[int(samples_number/2-0.1*samples_number):]*0 + des_x2+gamma_underbar, 
                         min_x2[int(samples_number/2-0.1*samples_number):]*0 + des_x2-gamma_underbar, facecolor='green', alpha=0.2, interpolate=True, label='$\epsilon$-stability bound')
        plt.scatter(x_axis_scale[int(samples_number/2)], fault_location_y, s=200, c='red', label='$F_{%s}$ - fault injected' %act_faulty_test)
        plt.xlabel("Time [s]")
        plt.ylabel("Angular speed [rad/s]")
        plt.legend(loc='best')
        plt.grid()
        plt.savefig(final_dir_ + title_fig_ref, dpi=dpi_)
        plt.close(fig)
        
        
        perc = 0.2  # percentage of time plot before and after fault   
        title_fig_c = "Reference_dynamics_x2_bounds_" + str(perc) + "perc.png"
        t_from = int(samples_number/2-perc*samples_number)
        t_to= int(samples_number/2+perc*samples_number)
        fig = plt.figure()
        plt.plot(x_axis_scale[t_from:t_to], min_x2[t_from:t_to]*0 + des_x2, '--g', label='$x_2^\star$')  # reference
        plt.fill_between(x_axis_scale[t_from:t_to], min_x2[t_from:t_to], 
                         max_x2[t_from:t_to], facecolor='blue', alpha=0.5, interpolate=True, label=c1)
        plt.fill_between(x_axis_scale[t_from:t_to], min_x2[t_from:t_to]*0 + des_x2+gamma_underbar, 
                         min_x2[t_from:t_to]*0 + des_x2-gamma_underbar, facecolor='green', alpha=0.2, interpolate=True, label='$\epsilon$-stability bound')
        fault_location_y = (max_x2[int(samples_number/2)] - min_x2[int(samples_number/2)])/2 + min_x2[int(samples_number/2)]
        plt.scatter(x_axis_scale[int(samples_number/2)], fault_location_y, s=100, c='red', label='$F_{%s}$ - fault injected' %act_faulty_test)
        plt.legend(loc='best')
        plt.xlabel("Time [s]")
        plt.ylabel("Force [N]")
        plt.grid()
        plt.savefig(final_dir_ + title_fig_c, dpi=dpi_)
        plt.close(fig)

    
    # 3) Lyapunov value along the ANLC trajectories
    title_fig_v = "Lyapunov_value_bounds.png"
    fig = plt.figure()
    plt.fill_between(x_axis_scale, min_V, 
                     max_V, facecolor='blue', alpha=0.5, interpolate=True)
    plt.xlabel("Time [s]")
    plt.ylabel(None)
    plt.grid()
    plt.savefig(final_dir_ + title_fig_v, dpi=dpi_)
    plt.close(fig)
        

    # 3) Lyapunov value along the ANLC trajectories
    t_short = 2
    title_fig_v = "Lyapunov_value_bounds_t" + str(t_short) + ".png"
    fig = plt.figure()
    plt.fill_between(x_axis_scale[:int(samples_number/t_short)], min_V[:int(samples_number/t_short)], 
                     max_V[:int(samples_number/t_short)], facecolor='blue', alpha=0.5, interpolate=True)
    plt.xlabel("Time [s]")
    plt.ylabel(None)
    plt.grid()
    plt.savefig(final_dir_ + title_fig_v, dpi=dpi_)
    plt.close(fig)
    
    
    # 3) Lyapunov value along the ANLC trajectories
    t_short = 6
    title_fig_v = "Lyapunov_value_bounds_t" + str(t_short) + ".png"
    fig = plt.figure()
    plt.fill_between(x_axis_scale[:int(samples_number/t_short)], min_V[:int(samples_number/t_short)], 
                     max_V[:int(samples_number/t_short)], facecolor='blue', alpha=0.5, interpolate=True)
    plt.xlabel("Time [s]")
    plt.ylabel(None)
    plt.grid()
    plt.savefig(final_dir_ + title_fig_v, dpi=dpi_)
    plt.close(fig)
    
    
    if dynamic_fault_test:
        
        
        # Dynamics 1
        title_fig_ref = "v3_Reference_dynamics_x1_bounds.png"
        fig = plt.figure()
        plt.plot(x_axis_scale, min_x1*0 + des_x1, '--g', label='$x_1^\star$')  # reference
        fault_location_y = (max_x1[int(samples_number/2)] - min_x1[int(samples_number/2)])/2 + min_x1[int(samples_number/2)]
        plt.fill_between(x_axis_scale, min_x1, 
                         max_x1, facecolor='blue', alpha=0.5, interpolate=True, label=c1)
        plt.fill_between(x_axis_scale, min_x1*0 + des_x1+gamma_underbar, 
                         min_x1*0 + des_x1-gamma_underbar, facecolor='green', alpha=0.2, interpolate=True, label='$\epsilon$-stability bound')
        plt.scatter(x_axis_scale[int(samples_number/2)], fault_location_y, s=100, c='red', label='$F_{%s}$ - fault injected' %act_faulty_test)
        plt.xlabel("Time [s]")
        plt.ylabel("Surge speed [m/s]")
        plt.legend(loc='best')
        plt.grid()
        plt.savefig(final_dir_ + title_fig_ref, dpi=dpi_)
        plt.close(fig)
        
        # Dynamics 1 -- legend on the lower right
        title_fig_ref = "v3_Reference_dynamics_x1_bounds_legend_low.png"
        fig = plt.figure()
        plt.plot(x_axis_scale, min_x1*0 + des_x1, '--g', label='$x_1^\star$')  # reference
        fault_location_y = (max_x1[int(samples_number/2)] - min_x1[int(samples_number/2)])/2 + min_x1[int(samples_number/2)]
        plt.fill_between(x_axis_scale, min_x1, 
                         max_x1, facecolor='blue', alpha=0.5, interpolate=True, label=c1)
        plt.fill_between(x_axis_scale, min_x1*0 + des_x1+gamma_underbar, 
                         min_x1*0 + des_x1-gamma_underbar, facecolor='green', alpha=0.2, interpolate=True, label='$\epsilon$-stability bound')
        plt.scatter(x_axis_scale[int(samples_number/2)], fault_location_y, s=100, c='red', label='$F_{%s}$ - fault injected' %act_faulty_test)
        plt.xlabel("Time [s]")
        plt.ylabel("Surge speed [m/s]")
        plt.legend(loc='lower right')
        plt.grid()
        plt.savefig(final_dir_ + title_fig_ref, dpi=dpi_)
        plt.close(fig)
        

        title_fig_ref = "v3_comp_Reference_dynamics_x1_bounds_v2.png"
        fig = plt.figure()
        plt.plot(x_axis_scale, min_x1*0 + des_x1, '--g', label='$x_1^\star$')  # reference
        plt.fill_between(x_axis_scale, min_x1, 
                         max_x1, facecolor='blue', alpha=0.5, interpolate=True, label=c1)
                
        plt.fill_between(x_axis_scale, min_x1*0 + des_x1+gamma_underbar, 
                         min_x1*0 + des_x1-gamma_underbar, facecolor='green', alpha=0.2, interpolate=True, label='$\epsilon$-stability bound')
        
        fault_location_y = (max_x1[int(samples_number/2)] - min_x1[int(samples_number/2)])/2 + min_x1[int(samples_number/2)]
        plt.scatter(x_axis_scale[int(samples_number/2)], fault_location_y, s=100, c='red', label='$F_{%s}$ - fault injected' %act_faulty_test)
        plt.xlabel("Time [s]")
        plt.ylabel("Surge speed [m/s]")
        plt.legend(loc='best')
        plt.grid()
        plt.savefig(final_dir_ + title_fig_ref, dpi=dpi_)
        plt.close(fig)


        # Dynamics 2
        title_fig_ref = "Reference_dynamics_x2_bounds_v2.png"
        fig, ax = plt.subplots()
        plt.plot(x_axis_scale, min_x2*0 + des_x2, '--g', label='$x_2^\star$')  # reference
        fault_location_y = (max_x2[int(samples_number/2)] - min_x2[int(samples_number/2)])/2 + min_x2[int(samples_number/2)]
        plt.fill_between(x_axis_scale, min_x2, 
                         max_x2, facecolor='blue', alpha=0.5, interpolate=True, label=c1)
        plt.fill_between(x_axis_scale, min_x2*0 + des_x2+gamma_underbar, 
                         min_x2*0 + des_x2-gamma_underbar, facecolor='green', alpha=0.2, interpolate=True, label='$\epsilon$-stability bound')
        plt.scatter(x_axis_scale[int(samples_number/2)], fault_location_y, s=100, c='red', label='$F_{%s}$ - fault injected' %act_faulty_test)
        ax.yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
        plt.xlabel("Time [s]")
        plt.ylabel("Angular speed [rad/s]")
        plt.legend(loc='best')
        plt.grid()
        plt.savefig(final_dir_ + title_fig_ref, dpi=dpi_)
        plt.close(fig)
        
        
        title_fig_ref = "v3_comp_Reference_dynamics_x2_bounds_v2.png"
        fig, ax = plt.subplots()
        plt.plot(x_axis_scale, min_x2*0 + des_x2, '--g', label='$x_2^\star$')  # reference
        plt.fill_between(x_axis_scale, min_x2, 
                         max_x2, facecolor='blue', alpha=0.5, interpolate=True, label=c1)
        
        plt.fill_between(x_axis_scale, min_x2*0 + des_x2+gamma_underbar, 
                         min_x2*0 + des_x2-gamma_underbar, facecolor='green', alpha=0.2, interpolate=True, label='$\epsilon$-stability bound')
        
        fault_location_y = (max_x2[int(samples_number/2)] - min_x2[int(samples_number/2)])/2 + min_x2[int(samples_number/2)]
        plt.scatter(x_axis_scale[int(samples_number/2)], fault_location_y, s=100, c='red', label='$F_{%s}$ - fault injected' %act_faulty_test)
        
        plt.scatter(x_axis_scale[int(samples_number/2)], fault_location_y, s=100, c='red')
        ax.yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
        plt.xlabel("Time [s]")
        plt.ylabel("Angular speed [rad/s]")
        plt.legend(loc='best')
        plt.grid()
        plt.savefig(final_dir_ + title_fig_ref, dpi=dpi_)
        plt.close(fig)
        
        
        title_fig_ref = "v3_Reference_dynamics_x2_bounds.png"
        fig, ax = plt.subplots()
        plt.plot(x_axis_scale, min_x2*0 + des_x2, '--g', label='$x_2^\star$')  # reference
        fault_location_y = (max_x2[int(samples_number/2)] - min_x2[int(samples_number/2)])/2 + min_x2[int(samples_number/2)]
        plt.fill_between(x_axis_scale, min_x2, 
                         max_x2, facecolor='blue', alpha=0.5, interpolate=True, label=c1)
        plt.fill_between(x_axis_scale, min_x2*0 + des_x2+gamma_underbar, 
                         min_x2*0 + des_x2-gamma_underbar, facecolor='green', alpha=0.2, interpolate=True, label='$\epsilon$-stability bound')
        plt.scatter(x_axis_scale[int(samples_number/2)], fault_location_y, s=100, c='red', label='$F_{%s}$ - fault injected' %act_faulty_test)
        ax.yaxis.set_major_formatter(FormatStrFormatter('%.3f'))
        plt.xlabel("Time [s]")
        plt.ylabel("Angular speed [rad/s]")
        plt.legend(loc='best')
        plt.grid()
        plt.savefig(final_dir_ + title_fig_ref, dpi=dpi_)
        plt.close(fig)



    # 2D trajectories
    title_fig_ref = "v3_" + str(c1) + "_dynamics_bounds.png"
    fig, ax = plt.subplots()
    plt.gca().set_aspect('equal')
    plt.plot(des_x1, des_x2, 'green', marker="x", label='$x^\star$')  # reference

    plt.plot(min_x1, min_x2, 'blue', alpha=0.5,  label=c1)
    
    # Plot Valid region computed by dReal
    r = gamma_overbar
    theta = np.linspace(0,2*np.pi,50)
    xc = r*np.cos(theta)
    yc = r*np.sin(theta)
    ax.plot(des_x1+xc[:],des_x2+yc[:],'k',linestyle='--', linewidth=1 , label='Domain')
    
    r = gamma_underbar
    theta = np.linspace(0,2*np.pi,50)
    xc = r*np.cos(theta)
    yc = r*np.sin(theta)
    ax.plot(des_x1+xc[:],des_x2+yc[:],'red',linestyle='--', linewidth=1 , label='$\epsilon$')
    
    ax.yaxis.set_major_formatter(FormatStrFormatter('%.3f'))
    plt.xlabel("Surge speed [m/s]")
    plt.ylabel("Angular speed [rad/s]")
    plt.legend(loc='best')
    plt.grid()
    plt.savefig(final_dir_ + title_fig_ref, dpi=dpi_)
    plt.close(fig)
    

    # 2D trajectories - zoom 
    title_fig_ref = "v3_" + str(c1) + "_dynamics_bounds_zoom_min.png"
    fig, ax = plt.subplots()
    plt.gca().set_aspect('equal')
    plt.plot(des_x1, des_x2, 'green', marker="x", label='$x^\star$')  # reference

    plt.plot(min_x1, min_x2, 'blue', alpha=0.5,  label=c1)
    
    # Plot Valid region computed by dReal    
    r = gamma_underbar
    theta = np.linspace(0,2*np.pi,50)
    xc = r*np.cos(theta)
    yc = r*np.sin(theta)
    ax.plot(des_x1+xc[:],des_x2+yc[:],'red',linestyle='--', linewidth=1 , label='$\epsilon$')
    
    ax.yaxis.set_major_formatter(FormatStrFormatter('%.3f'))
    plt.xlabel("Surge speed [m/s]")
    plt.ylabel("Angular speed [rad/s]")
    plt.legend(loc='best')
    plt.grid()
    plt.savefig(final_dir_ + title_fig_ref, dpi=dpi_)
    plt.close(fig)


    # 2D trajectories - zoom 
    title_fig_ref = "v3_" + str(c1) + "_dynamics_bounds_zoom_max.png"
    fig, ax = plt.subplots()
    plt.gca().set_aspect('equal')
    plt.plot(des_x1, des_x2, 'green', marker="x", label='$x^\star$')  # reference

    plt.plot(max_x1, max_x2, 'blue', alpha=0.5,  label=c1)

    
    # Plot Valid region computed by dReal    
    r = gamma_underbar
    theta = np.linspace(0,2*np.pi,50)
    xc = r*np.cos(theta)
    yc = r*np.sin(theta)
    ax.plot(des_x1+xc[:],des_x2+yc[:],'red',linestyle='--', linewidth=1 , label='$\epsilon$')
    
    ax.yaxis.set_major_formatter(FormatStrFormatter('%.3f'))
    plt.xlabel("Surge speed [m/s]")
    plt.ylabel("Angular speed [rad/s]")
    plt.legend(loc='best')
    plt.grid()
    plt.savefig(final_dir_ + title_fig_ref, dpi=dpi_)
    plt.close(fig)


    # saving performance report
    range_x1 = max(max_x1) - min(min_x1)
    range_x2 = max(max_x2) - min(min_x2)
    ideal_x1 = des_x1 - init_x1
    ideal_x2 = des_x2 - init_x2
    
    perfo_report = [f"Max x1 = {max(max_x1)}\
                    \nMin x1 = {min(min_x1)}\
                    \nOvershoot x1 % = {(range_x1-ideal_x1)/ideal_x1*100}%\
                    \nMax x2 = {max(max_x2)}\
                    \nMin x2 = {min(min_x2)}\
                    \nOvershoot x2 % = {(range_x2-ideal_x2)/ideal_x2*100}%\
                    \nMax(abs(u1)) = {max(abs(max_u1))}\
                    \nMax(abs(u2)) = {max(abs(max_u2))}\
                    \nMax(abs(u3)) = {max(abs(max_u3))}\
                    "]
    np.savetxt(final_dir_ + "perfo_report.txt", perfo_report, fmt="%s")


def plot_comparison(x_all, u_all, V_all, x_axis_scale, 
                    dynamic_fault_test, c1, c6, act_faulty_test,
                    des_x1, des_x2, init_x1, init_x2, gamma_underbar, 
                    samples_number, ann_files, 
                    x_all_lqr, u_all_lqr,
                    dpi_, final_dir_):  # run_statistics
    
    # This function plots the bounds of the pFT-ANLC controllers and the bounds of the 
    # state-space controller

    # Obtaining boundaries pFT-ANLC
    min_x1 = x_all[0][:,0].copy()
    min_x2 = x_all[0][:,1].copy()
    min_u1 = u_all[0][:,0].copy()
    min_u2 = u_all[0][:,1].copy()
    min_u3 = u_all[0][:,2].copy()
    min_V = V_all[0][:,0].copy()
   
    max_x1 = x_all[0][:,0].copy()
    max_x2 = x_all[0][:,1].copy()
    max_u1 = u_all[0][:,0].copy()
    max_u2 = u_all[0][:,1].copy()
    max_u3 = u_all[0][:,2].copy()
    max_V = V_all[0][:,0].copy()
    
    for jCntr in range(ann_files):
        min_x1 = np.minimum(min_x1, x_all[jCntr][:,0]).copy()
        min_x2 = np.minimum(min_x2, x_all[jCntr][:,1]).copy()
        min_u1 = np.minimum(min_u1, u_all[jCntr][:,0]).copy()
        min_u2 = np.minimum(min_u2, u_all[jCntr][:,1]).copy()
        min_u3 = np.minimum(min_u3, u_all[jCntr][:,2]).copy()
        min_V = np.minimum(min_V, V_all[jCntr][:,0]).copy()

        max_x1 = np.maximum(max_x1, x_all[jCntr][:,0]).copy()
        max_x2 = np.maximum(max_x2, x_all[jCntr][:,1]).copy()
        max_u1 = np.maximum(max_u1, u_all[jCntr][:,0]).copy()
        max_u2 = np.maximum(max_u2, u_all[jCntr][:,1]).copy()
        max_u3 = np.maximum(max_u3, u_all[jCntr][:,2]).copy()
        max_V = np.maximum(max_V, V_all[jCntr][:,0]).copy()
    

    # Obtaining boundaries controller state-space
    min_x1_lqr = x_all_lqr[0][:,0].copy()
    min_x2_lqr = x_all_lqr[0][:,1].copy()
    min_u1_lqr = u_all_lqr[0][:,0].copy()
    min_u2_lqr = u_all_lqr[0][:,1].copy()
    min_u3_lqr = u_all_lqr[0][:,2].copy()
   
    max_x1_lqr = x_all_lqr[0][:,0].copy()
    max_x2_lqr = x_all_lqr[0][:,1].copy()
    max_u1_lqr = u_all_lqr[0][:,0].copy()
    max_u2_lqr = u_all_lqr[0][:,1].copy()
    max_u3_lqr = u_all_lqr[0][:,2].copy()
    

    for jCntr in range(2):
        min_x1_lqr = np.minimum(min_x1_lqr, x_all_lqr[jCntr][:,0]).copy()
        min_x2_lqr = np.minimum(min_x2_lqr, x_all_lqr[jCntr][:,1]).copy()
        min_u1_lqr = np.minimum(min_u1_lqr, u_all_lqr[jCntr][:,0]).copy()
        min_u2_lqr = np.minimum(min_u2_lqr, u_all_lqr[jCntr][:,1]).copy()
        min_u3_lqr = np.minimum(min_u3_lqr, u_all_lqr[jCntr][:,2]).copy()

        max_x1_lqr = np.maximum(max_x1_lqr, x_all_lqr[jCntr][:,0]).copy()
        max_x2_lqr = np.maximum(max_x2_lqr, x_all_lqr[jCntr][:,1]).copy()
        max_u1_lqr = np.maximum(max_u1_lqr, u_all_lqr[jCntr][:,0]).copy()
        max_u2_lqr = np.maximum(max_u2_lqr, u_all_lqr[jCntr][:,1]).copy()
        max_u3_lqr = np.maximum(max_u3_lqr, u_all_lqr[jCntr][:,2]).copy()
   

    # 1.1 - all forces together
    title_fig_c = "Control_input_forces_bounds_" + str(c1) +".png"
    fig = plt.figure()
    plt.fill_between(x_axis_scale, min_u1, max_u1, facecolor='blue', alpha=0.8, interpolate=True, label='$F_1$')
    plt.fill_between(x_axis_scale, min_u2, max_u2, facecolor='red', alpha=0.8, interpolate=True, label='$F_2$')
    plt.fill_between(x_axis_scale, min_u3, max_u3, facecolor='green', alpha=0.8, interpolate=True, label='$F_3$')
    plt.legend(loc='best')
    plt.xlabel("Time [s]")
    plt.ylabel("Force [N]")
    plt.grid()
    plt.savefig(final_dir_ + title_fig_c, dpi=dpi_)
    plt.close(fig)
    
    
    if dynamic_fault_test:
        # 1.1 - all forces together
        title_fig_c = "Control_input_forces_bounds_v2_" + str(c1) +".png"
        fig = plt.figure()
        
        plt.fill_between(x_axis_scale, min_u1, max_u1, facecolor='blue', alpha=0.8, interpolate=True, label='$F_1$')
        plt.fill_between(x_axis_scale, min_u2, max_u2, facecolor='red', alpha=0.8, interpolate=True, label='$F_2$')
        plt.fill_between(x_axis_scale, min_u3, max_u3, facecolor='green', alpha=0.8, interpolate=True, label='$F_3$')
        
        if act_faulty_test==1:
            fault_location_y = (max_u1[int(samples_number/2)] - min_u1[int(samples_number/2)])/2 + min_u1[int(samples_number/2)]
        if act_faulty_test==2:
            fault_location_y = (max_u2[int(samples_number/2)] - min_u2[int(samples_number/2)])/2 + min_u2[int(samples_number/2)]
        if act_faulty_test==3:
            fault_location_y = (max_u3[int(samples_number/2)] - min_u3[int(samples_number/2)])/2 + min_u3[int(samples_number/2)]
        
        plt.scatter(x_axis_scale[int(samples_number/2)], fault_location_y, s=100, c='red', label='$F_{%s}$ - fault injected' %act_faulty_test)
        plt.legend(loc='best')
        plt.xlabel("Time [s]")
        plt.ylabel("Force [N]")
        plt.grid()
        plt.savefig(final_dir_ + title_fig_c, dpi=dpi_)
        plt.close(fig)
    
    
    # 1.1 - all forces together - half time 
    t_short = 2
    title_fig_c = "Control_input_forces_bounds_t" + str(t_short) + "_" + str(c1) +".png"
    fig = plt.figure()
    plt.fill_between(x_axis_scale[:int(samples_number/t_short)], min_u1[:int(samples_number/t_short)], max_u1[:int(samples_number/t_short)], facecolor='blue', alpha=0.8, interpolate=True, label='$F_1$')
    plt.fill_between(x_axis_scale[:int(samples_number/t_short)], min_u2[:int(samples_number/t_short)], max_u2[:int(samples_number/t_short)], facecolor='red', alpha=0.8, interpolate=True, label='$F_2$')
    plt.fill_between(x_axis_scale[:int(samples_number/t_short)], min_u3[:int(samples_number/t_short)], max_u3[:int(samples_number/t_short)], facecolor='green', alpha=0.8, interpolate=True, label='$F_3$')
    plt.legend(loc='best')
    plt.xlabel("Time [s]")
    plt.ylabel("Force [N]")
    plt.grid()
    plt.savefig(final_dir_ + title_fig_c, dpi=dpi_)
    plt.close(fig)
    

    t_short = 6
    title_fig_c = "Control_input_forces_bounds_t" + str(t_short) + "_" + str(c1) +".png"
    fig = plt.figure()
    plt.fill_between(x_axis_scale[:int(samples_number/t_short)], min_u1[:int(samples_number/t_short)], max_u1[:int(samples_number/t_short)], facecolor='blue', alpha=0.8, interpolate=True, label='$F_1$')
    plt.fill_between(x_axis_scale[:int(samples_number/t_short)], min_u2[:int(samples_number/t_short)], max_u2[:int(samples_number/t_short)], facecolor='red', alpha=0.8, interpolate=True, label='$F_2$')
    plt.fill_between(x_axis_scale[:int(samples_number/t_short)], min_u3[:int(samples_number/t_short)], max_u3[:int(samples_number/t_short)], facecolor='green', alpha=0.8, interpolate=True, label='$F_3$')
    plt.legend(loc='best')
    plt.xlabel("Time [s]")
    plt.ylabel("Force [N]")
    plt.grid()
    plt.savefig(final_dir_ + title_fig_c, dpi=dpi_)
    plt.close(fig)


    if dynamic_fault_test:
        # 1.1 - all forces together (final section)
        title_fig_c = "Control_input_forces_short_bounds_" + str(c1) +".png"
        fig = plt.figure()
        plt.fill_between(x_axis_scale[int(samples_number/2-0.1*samples_number):], min_u1[int(samples_number/2-0.1*samples_number):], max_u1[int(samples_number/2-0.1*samples_number):], facecolor='blue', alpha=0.8, interpolate=True, label='$F_1$')
        plt.fill_between(x_axis_scale[int(samples_number/2-0.1*samples_number):], min_u2[int(samples_number/2-0.1*samples_number):], max_u2[int(samples_number/2-0.1*samples_number):], facecolor='red', alpha=0.8, interpolate=True, label='$F_2$')
        plt.fill_between(x_axis_scale[int(samples_number/2-0.1*samples_number):], min_u3[int(samples_number/2-0.1*samples_number):], max_u3[int(samples_number/2-0.1*samples_number):], facecolor='green', alpha=0.8, interpolate=True, label='$F_3$')
        plt.legend(loc='best')
        plt.xlabel("Time [s]")
        plt.ylabel("Force [N]")
        plt.grid()
        plt.savefig(final_dir_ + title_fig_c, dpi=dpi_)
        plt.close(fig)


        
    title_fig_ref = "Reference_dynamics_x1_bounds_" + str(c1) +".png"
    fig = plt.figure()
    plt.plot(x_axis_scale, min_x1*0 + des_x1, '--g', label='$x_1^\star$')  # reference
    plt.fill_between(x_axis_scale, min_x1, 
                     max_x1, facecolor='blue', alpha=0.5, interpolate=True, label=c1)
    plt.fill_between(x_axis_scale, min_x1*0 + des_x1+gamma_underbar, 
                     min_x1*0 + des_x1-gamma_underbar, facecolor='green', alpha=0.2, interpolate=True, label='$\epsilon$-stability bound')
    plt.xlabel("Time [s]")
    plt.ylabel("Surge speed [m/s]")
    plt.legend(loc='best')
    plt.grid()
    plt.savefig(final_dir_ + title_fig_ref, dpi=dpi_)
    plt.close(fig)


    t_short = 2
    title_fig_ref = "Reference_dynamics_x1_bounds_t" + str(t_short) + "_" + str(c1) +".png"
    fig = plt.figure()
    plt.plot(x_axis_scale[:int(samples_number/t_short)], min_x1[:int(samples_number/t_short)]*0 + des_x1, '--g', label='$x_1^\star$')  # reference
    plt.fill_between(x_axis_scale[:int(samples_number/t_short)], min_x1[:int(samples_number/t_short)], 
                     max_x1[:int(samples_number/t_short)], facecolor='blue', alpha=0.5, interpolate=True, label=c1)
    plt.fill_between(x_axis_scale[:int(samples_number/t_short)], min_x1[:int(samples_number/t_short)]*0 + des_x1+gamma_underbar, 
                     min_x1[:int(samples_number/t_short)]*0 + des_x1-gamma_underbar, facecolor='green', alpha=0.2, interpolate=True, label='$\epsilon$-stability bound')
    plt.xlabel("Time [s]")
    plt.ylabel("Surge speed [m/s]")
    plt.legend(loc='best')
    plt.grid()
    plt.savefig(final_dir_ + title_fig_ref, dpi=dpi_)
    plt.close(fig)


    t_short = 6
    title_fig_ref = "Reference_dynamics_x1_bounds_t" + str(t_short) + "_" + str(c1) +".png"
    fig = plt.figure()
    plt.plot(x_axis_scale[:int(samples_number/t_short)], min_x1[:int(samples_number/t_short)]*0 + des_x1, '--g', label='$x_{1*}$')  # reference
    plt.fill_between(x_axis_scale[:int(samples_number/t_short)], min_x1[:int(samples_number/t_short)], 
                     max_x1[:int(samples_number/t_short)], facecolor='blue', alpha=0.5, interpolate=True, label=c1)
    plt.fill_between(x_axis_scale[:int(samples_number/t_short)], min_x1[:int(samples_number/t_short)]*0 + des_x1+gamma_underbar, 
                     min_x1[:int(samples_number/t_short)]*0 + des_x1-gamma_underbar, facecolor='green', alpha=0.2, interpolate=True, label='$\epsilon$-stability bound')
    plt.xlabel("Time [s]")
    plt.ylabel("Surge speed [m/s]")
    plt.legend(loc='best')
    plt.grid()
    plt.savefig(final_dir_ + title_fig_ref, dpi=dpi_)
    plt.close(fig)


    if dynamic_fault_test:
        
        title_fig_ref = "Reference_dynamics_x1_bounds_short_" + str(c1) +".png"
        fig = plt.figure()
        plt.plot(x_axis_scale[int(samples_number/2-0.1*samples_number):], min_x1[int(samples_number/2-0.1*samples_number):]*0 + des_x1, '--g', label='$x_1^\star$')  # reference
        plt.fill_between(x_axis_scale[int(samples_number/2-0.1*samples_number):], min_x1[int(samples_number/2-0.1*samples_number):], 
                         max_x1[int(samples_number/2-0.1*samples_number):], facecolor='blue', alpha=0.5, interpolate=True, label=c1)
        plt.fill_between(x_axis_scale[int(samples_number/2-0.1*samples_number):], min_x1[int(samples_number/2-0.1*samples_number):]*0 + des_x1+gamma_underbar, 
                         min_x1[int(samples_number/2-0.1*samples_number):]*0 + des_x1-gamma_underbar, facecolor='green', alpha=0.2, interpolate=True, label='$\epsilon$-stability bound')
        plt.xlabel("Time [s]")
        plt.ylabel("Surge speed [m/s]")
        plt.legend(loc='best')
        plt.grid()
        plt.savefig(final_dir_ + title_fig_ref, dpi=dpi_)
        plt.close(fig)


        fault_location_y = (max_x1[int(samples_number/2)] - min_x1[int(samples_number/2)])/2 + min_x1[int(samples_number/2)]
        title_fig_ref = "Reference_dynamics_x1_bounds_short_v2_" + str(c1) +".png"
        fig = plt.figure()
        plt.plot(x_axis_scale[int(samples_number/2-0.1*samples_number):], min_x1[int(samples_number/2-0.1*samples_number):]*0 + des_x1, '--g', label='$x_1^\star$')  # reference
        plt.fill_between(x_axis_scale[int(samples_number/2-0.1*samples_number):], min_x1[int(samples_number/2-0.1*samples_number):], 
                         max_x1[int(samples_number/2-0.1*samples_number):], facecolor='blue', alpha=0.5, interpolate=True, label=c1)
        plt.fill_between(x_axis_scale[int(samples_number/2-0.1*samples_number):], min_x1[int(samples_number/2-0.1*samples_number):]*0 + des_x1+gamma_underbar, 
                         min_x1[int(samples_number/2-0.1*samples_number):]*0 + des_x1-gamma_underbar, facecolor='green', alpha=0.2, interpolate=True, label='$\epsilon$-stability bound')
        plt.scatter(x_axis_scale[int(samples_number/2)], fault_location_y, s=200, c='red', label='$F_{%s}$ - fault injected' %act_faulty_test)
        plt.xlabel("Time [s]")
        plt.ylabel("Surge speed [m/s]")
        plt.legend(loc='best')
        plt.grid()
        plt.savefig(final_dir_ + title_fig_ref, dpi=dpi_)
        plt.close(fig)


    # 4) Dynamics x2
    title_fig_ref = "Reference_dynamics_x2_bounds_" + str(c1) +".png"
    fig, ax = plt.subplots()
    plt.plot(x_axis_scale, min_x2*0 + des_x2, '--g', label='$x_2^\star$')  # reference
    plt.fill_between(x_axis_scale, min_x2, 
                     max_x2, facecolor='blue', alpha=0.5, interpolate=True, label=c1)
    plt.fill_between(x_axis_scale, min_x2*0 + des_x2+gamma_underbar, 
                     min_x2*0 + des_x2-gamma_underbar, facecolor='green', alpha=0.2, interpolate=True, label='$\epsilon$-stability bound')
    ax.yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
    plt.xlabel("Time [s]")
    plt.ylabel("Angular speed [rad/s]")
    plt.legend(loc='best')
    plt.grid()
    plt.savefig(final_dir_ + title_fig_ref, dpi=dpi_)
    plt.close(fig)


    # 4) Dynamics x2
    t_short = 2
    title_fig_ref = "Reference_dynamics_x2_bounds_t" + str(t_short) + "_" + str(c1) +".png"
    fig, ax = plt.subplots()
    plt.plot(x_axis_scale[:int(samples_number/t_short)], min_x2[:int(samples_number/t_short)]*0 + des_x2, '--g', label='$x_2^\star$')  # reference
    plt.fill_between(x_axis_scale[:int(samples_number/t_short)], min_x2[:int(samples_number/t_short)], 
                     max_x2[:int(samples_number/t_short)], facecolor='blue', alpha=0.5, interpolate=True, label=c1)
    plt.fill_between(x_axis_scale[:int(samples_number/t_short)], min_x2[:int(samples_number/t_short)]*0 + des_x2+gamma_underbar, 
                     min_x2[:int(samples_number/t_short)]*0 + des_x2-gamma_underbar, facecolor='green', alpha=0.2, interpolate=True, label='$\epsilon$-stability bound')
    ax.yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
    plt.xlabel("Time [s]")
    plt.ylabel("Angular speed [rad/s]")
    plt.legend(loc='best')
    plt.grid()
    plt.savefig(final_dir_ + title_fig_ref, dpi=dpi_)
    plt.close(fig)


    # 4) Dynamics x2
    t_short = 6
    title_fig_ref = "Reference_dynamics_x2_bounds_t" + str(t_short) + "_" + str(c1) +".png"
    fig, ax = plt.subplots()
    plt.plot(x_axis_scale[:int(samples_number/t_short)], min_x2[:int(samples_number/t_short)]*0 + des_x2, '--g', label='$x_2^\star$')  # reference
    plt.fill_between(x_axis_scale[:int(samples_number/t_short)], min_x2[:int(samples_number/t_short)], 
                     max_x2[:int(samples_number/t_short)], facecolor='blue', alpha=0.5, interpolate=True, label=c1)
    plt.fill_between(x_axis_scale[:int(samples_number/t_short)], min_x2[:int(samples_number/t_short)]*0 + des_x2+gamma_underbar, 
                     min_x2[:int(samples_number/t_short)]*0 + des_x2-gamma_underbar, facecolor='green', alpha=0.2, interpolate=True, label='$\epsilon$-stability bound')
    ax.yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
    plt.xlabel("Time [s]")
    plt.ylabel("Angular speed [rad/s]")
    plt.legend(loc='best')
    plt.grid()
    plt.savefig(final_dir_ + title_fig_ref, dpi=dpi_)
    plt.close(fig)


    if dynamic_fault_test:
        
        title_fig_ref = "Reference_dynamics_x2_short_bounds_"+ str(c1) +".png"
        fig, ax = plt.subplots()
        plt.plot(x_axis_scale[int(samples_number/2-0.1*samples_number):], min_x2[int(samples_number/2-0.1*samples_number):]*0 + des_x2, '--g', label='$x_2^\star$')  # reference
        plt.fill_between(x_axis_scale[int(samples_number/2-0.1*samples_number):], min_x2[int(samples_number/2-0.1*samples_number):], 
                         max_x2[int(samples_number/2-0.1*samples_number):], facecolor='blue', alpha=0.5, interpolate=True, label=c1)
        plt.fill_between(x_axis_scale[int(samples_number/2-0.1*samples_number):], min_x2[int(samples_number/2-0.1*samples_number):]*0 + des_x2+gamma_underbar, 
                         min_x2[int(samples_number/2-0.1*samples_number):]*0 + des_x2-gamma_underbar, facecolor='green', alpha=0.2, interpolate=True, label='$\epsilon$-stability bound')
        ax.yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
        plt.xlabel("Time [s]")
        plt.ylabel("Angular speed [rad/s]")
        plt.legend(loc='best')
        plt.grid()
        plt.savefig(final_dir_ + title_fig_ref, dpi=dpi_)
        plt.close(fig)


        fault_location_y = (max_x2[int(samples_number/2)] - min_x2[int(samples_number/2)])/2 + min_x2[int(samples_number/2)]
        title_fig_ref = "Reference_dynamics_x2_short_v2_bounds_"+ str(c1) +".png"
        fig, ax = plt.subplots()
        plt.plot(x_axis_scale[int(samples_number/2-0.1*samples_number):], min_x2[int(samples_number/2-0.1*samples_number):]*0 + des_x2, '--g', label='$x_2^\star$')  # reference
        plt.fill_between(x_axis_scale[int(samples_number/2-0.1*samples_number):], min_x2[int(samples_number/2-0.1*samples_number):], 
                         max_x2[int(samples_number/2-0.1*samples_number):], facecolor='blue', alpha=0.5, interpolate=True, label=c1)
        plt.fill_between(x_axis_scale[int(samples_number/2-0.1*samples_number):], min_x2[int(samples_number/2-0.1*samples_number):]*0 + des_x2+gamma_underbar, 
                         min_x2[int(samples_number/2-0.1*samples_number):]*0 + des_x2-gamma_underbar, facecolor='green', alpha=0.2, interpolate=True, label='$\epsilon$-stability bound')
        plt.scatter(x_axis_scale[int(samples_number/2)], fault_location_y, s=200, c='red', label='$F_{%s}$ - fault injected' %act_faulty_test)
        ax.yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
        plt.xlabel("Time [s]")
        plt.ylabel("Angular speed [rad/s]")
        plt.legend(loc='best')
        plt.grid()
        plt.savefig(final_dir_ + title_fig_ref, dpi=dpi_)
        plt.close(fig)

        
    # 3) Lyapunov value along the ANLC trajectories
    title_fig_v = "Lyapunov_value_bounds_"+ str(c1) +".png"
    fig = plt.figure()
    plt.fill_between(x_axis_scale, min_V, 
                     max_V, facecolor='blue', alpha=0.5, interpolate=True)
    plt.xlabel("Time [s]")
    plt.ylabel(None)
    plt.grid()
    plt.savefig(final_dir_ + title_fig_v, dpi=dpi_)
    plt.close(fig)
        

    # 3) Lyapunov value along the ANLC trajectories
    t_short = 2
    title_fig_v = "Lyapunov_value_bounds_t" + str(t_short) + "_" + str(c1) +".png"
    fig = plt.figure()
    plt.fill_between(x_axis_scale[:int(samples_number/t_short)], min_V[:int(samples_number/t_short)], 
                     max_V[:int(samples_number/t_short)], facecolor='blue', alpha=0.5, interpolate=True)
    plt.xlabel("Time [s]")
    plt.ylabel(None)
    plt.grid()
    plt.savefig(final_dir_ + title_fig_v, dpi=dpi_)
    plt.close(fig)
    
    
    # 3) Lyapunov value along the ANLC trajectories
    t_short = 6
    title_fig_v = "Lyapunov_value_bounds_t" + str(t_short) + "_" + str(c1) +".png"
    fig = plt.figure()
    plt.fill_between(x_axis_scale[:int(samples_number/t_short)], min_V[:int(samples_number/t_short)], 
                     max_V[:int(samples_number/t_short)], facecolor='blue', alpha=0.5, interpolate=True)
    plt.xlabel("Time [s]")
    plt.ylabel(None)
    plt.grid()
    plt.savefig(final_dir_ + title_fig_v, dpi=dpi_)
    plt.close(fig)
    
    
    
    # saving performance report
    range_x1 = max(max_x1) - min(min_x1)
    range_x2 = max(max_x2) - min(min_x2)
    ideal_x1 = des_x1 - init_x1
    ideal_x2 = des_x2 - init_x2
    
    perfo_report = [f"Max x1 = {max(max_x1)}\
                    \nMin x1 = {min(min_x1)}\
                    \nOvershoot x1 % = {(range_x1-ideal_x1)/ideal_x1*100}%\
                    \nMax x2 = {max(max_x2)}\
                    \nMin x2 = {min(min_x2)}\
                    \nOvershoot x2 % = {(range_x2-ideal_x2)/ideal_x2*100}%\
                    \nMax(abs(u1)) = {max(abs(max_u1))}\
                    \nMax(abs(u2)) = {max(abs(max_u2))}\
                    \nMax(abs(u3)) = {max(abs(max_u3))}\
                    "]
    np.savetxt(final_dir_ + "perfo_report.txt", perfo_report, fmt="%s")


    # COMPARISON
    t_short = 1
    title_fig_ref = "comp_Reference_dynamics_x1_bounds_t" + str(t_short) + "_comp.png"
    fig = plt.figure()
    plt.plot(x_axis_scale[:int(samples_number/t_short)], min_x1[:int(samples_number/t_short)]*0 + des_x1, '--g', label='$x_1^\star$')  # reference
    
    plt.fill_between(x_axis_scale[:int(samples_number/t_short)], min_x1[:int(samples_number/t_short)], 
                     max_x1[:int(samples_number/t_short)], facecolor='blue', alpha=0.5, interpolate=True, label=c1)
    
    plt.fill_between(x_axis_scale[:int(samples_number/t_short)], min_x1_lqr[:int(samples_number/t_short)], 
                     max_x1_lqr[:int(samples_number/t_short)], facecolor='purple', alpha=0.5, interpolate=True, label=c6)
    
    plt.fill_between(x_axis_scale[:int(samples_number/t_short)], min_x1[:int(samples_number/t_short)]*0 + des_x1+gamma_underbar, 
                     min_x1[:int(samples_number/t_short)]*0 + des_x1-gamma_underbar, facecolor='green', alpha=0.2, interpolate=True, label='$\epsilon$-stability bound')
    
    plt.xlabel("Time [s]")
    plt.ylabel("Surge speed [m/s]")
    plt.legend(loc='best')
    plt.grid()
    plt.savefig(final_dir_ + title_fig_ref, dpi=dpi_)
    plt.close(fig)

    
    t_short = 2
    title_fig_ref = "comp_Reference_dynamics_x1_bounds_t" + str(t_short) + "_comp.png"
    fig = plt.figure()
    plt.plot(x_axis_scale[:int(samples_number/t_short)], min_x1[:int(samples_number/t_short)]*0 + des_x1, '--g', label='$x_1^\star$')  # reference
    
    plt.fill_between(x_axis_scale[:int(samples_number/t_short)], min_x1[:int(samples_number/t_short)], 
                     max_x1[:int(samples_number/t_short)], facecolor='blue', alpha=0.5, interpolate=True, label=c1)
    
    plt.fill_between(x_axis_scale[:int(samples_number/t_short)], min_x1_lqr[:int(samples_number/t_short)], 
                     max_x1_lqr[:int(samples_number/t_short)], facecolor='purple', alpha=0.5, interpolate=True, label=c6)
    
    plt.fill_between(x_axis_scale[:int(samples_number/t_short)], min_x1[:int(samples_number/t_short)]*0 + des_x1+gamma_underbar, 
                     min_x1[:int(samples_number/t_short)]*0 + des_x1-gamma_underbar, facecolor='green', alpha=0.2, interpolate=True, label='$\epsilon$-stability bound')
    
    plt.xlabel("Time [s]")
    plt.ylabel("Surge speed [m/s]")
    plt.legend(loc='best')
    plt.grid()
    plt.savefig(final_dir_ + title_fig_ref, dpi=dpi_)
    plt.close(fig)


    t_short = 4
    title_fig_ref = "comp_Reference_dynamics_x1_bounds_t" + str(t_short) + "_comp.png"
    fig = plt.figure()
    plt.plot(x_axis_scale[:int(samples_number/t_short)], min_x1[:int(samples_number/t_short)]*0 + des_x1, '--g', label='$x_1^\star$')  # reference
    
    plt.fill_between(x_axis_scale[:int(samples_number/t_short)], min_x1[:int(samples_number/t_short)], 
                     max_x1[:int(samples_number/t_short)], facecolor='blue', alpha=0.5, interpolate=True, label=c1)
    
    plt.fill_between(x_axis_scale[:int(samples_number/t_short)], min_x1_lqr[:int(samples_number/t_short)], 
                     max_x1_lqr[:int(samples_number/t_short)], facecolor='purple', alpha=0.5, interpolate=True, label=c6)
    
    plt.fill_between(x_axis_scale[:int(samples_number/t_short)], min_x1[:int(samples_number/t_short)]*0 + des_x1+gamma_underbar, 
                     min_x1[:int(samples_number/t_short)]*0 + des_x1-gamma_underbar, facecolor='green', alpha=0.2, interpolate=True, label='$\epsilon$-stability bound')
    
    plt.xlabel("Time [s]")
    plt.ylabel("Surge speed [m/s]")
    plt.legend(loc='best')
    plt.grid()
    plt.savefig(final_dir_ + title_fig_ref, dpi=dpi_)
    plt.close(fig)



    # FORCE 1
    title_fig_c = "comp_Control_input_forces_bounds_F1.png"
    fig = plt.figure()
    plt.fill_between(x_axis_scale, min_u1, max_u1, facecolor='red', alpha=0.8, interpolate=True, label=c1)
    plt.fill_between(x_axis_scale, min_u1_lqr, max_u1_lqr, facecolor='blue', alpha=0.8, interpolate=True, label=c6)
    plt.legend(loc='best')
    plt.xlabel("Time [s]")
    plt.ylabel("Force [N]")
    plt.grid()
    plt.savefig(final_dir_ + title_fig_c, dpi=dpi_)
    plt.close(fig)

    # FORCE 2
    title_fig_c = "comp_Control_input_forces_bounds_F2.png"
    fig = plt.figure()
    plt.fill_between(x_axis_scale, min_u2, max_u2, facecolor='red', alpha=0.8, interpolate=True, label=c1)
    plt.fill_between(x_axis_scale, min_u2_lqr, max_u2_lqr, facecolor='blue', alpha=0.8, interpolate=True, label=c6)
    plt.legend(loc='best')
    plt.xlabel("Time [s]")
    plt.ylabel("Force [N]")
    plt.grid()
    plt.savefig(final_dir_ + title_fig_c, dpi=dpi_)
    plt.close(fig)


    # FORCE 3
    title_fig_c = "comp_Control_input_forces_bounds_F3.png"
    fig = plt.figure()
    #plt.fill_between(x_axis_scale, min_u1, max_u1, facecolor='blue', alpha=0.8, interpolate=True, label=c1)
    plt.fill_between(x_axis_scale, min_u3, max_u3, facecolor='red', alpha=0.8, interpolate=True, label=c1)
    plt.fill_between(x_axis_scale, min_u3_lqr, max_u3_lqr, facecolor='blue', alpha=0.8, interpolate=True, label=c6)
    plt.legend(loc='best')
    plt.xlabel("Time [s]")
    plt.ylabel("Force [N]")
    plt.grid()
    plt.savefig(final_dir_ + title_fig_c, dpi=dpi_)
    plt.close(fig)


    if dynamic_fault_test:

        title_fig_ref = "comp_Reference_dynamics_x1_bounds_short.png"
        fig = plt.figure()
        plt.plot(x_axis_scale[int(samples_number/2-0.1*samples_number):], min_x1[int(samples_number/2-0.1*samples_number):]*0 + des_x1, '--g', label='$x_1^\star$')  # reference
        plt.fill_between(x_axis_scale[int(samples_number/2-0.1*samples_number):], min_x1[int(samples_number/2-0.1*samples_number):], 
                         max_x1[int(samples_number/2-0.1*samples_number):], facecolor='blue', alpha=0.5, interpolate=True, label=c1)
        
        plt.fill_between(x_axis_scale[int(samples_number/2-0.1*samples_number):], min_x1_lqr[int(samples_number/2-0.1*samples_number):], 
                         max_x1_lqr[int(samples_number/2-0.1*samples_number):], facecolor='purple', alpha=0.5, interpolate=True, label=c6)        
        
        plt.fill_between(x_axis_scale[int(samples_number/2-0.1*samples_number):], min_x1[int(samples_number/2-0.1*samples_number):]*0 + des_x1+gamma_underbar, 
                         min_x1[int(samples_number/2-0.1*samples_number):]*0 + des_x1-gamma_underbar, facecolor='green', alpha=0.2, interpolate=True, label='$\epsilon$-stability bound')
        plt.xlabel("Time [s]")
        plt.ylabel("Surge speed [m/s]")
        plt.legend(loc='best')
        plt.grid()
        plt.savefig(final_dir_ + title_fig_ref, dpi=dpi_)
        plt.close(fig)

    if dynamic_fault_test:

        # Dynamics 1
        title_fig_ref = "v3_Reference_dynamics_x1_bounds.png"
        fig = plt.figure()
        plt.plot(x_axis_scale, min_x1*0 + des_x1, '--g', label='$x_1^\star$')  # reference
        fault_location_y = (max_x1[int(samples_number/2)] - min_x1[int(samples_number/2)])/2 + min_x1[int(samples_number/2)]
        plt.fill_between(x_axis_scale, min_x1, 
                         max_x1, facecolor='blue', alpha=0.5, interpolate=True, label=c1)
        plt.fill_between(x_axis_scale, min_x1*0 + des_x1+gamma_underbar, 
                         min_x1*0 + des_x1-gamma_underbar, facecolor='green', alpha=0.2, interpolate=True, label='$\epsilon$-stability bound')
        plt.scatter(x_axis_scale[int(samples_number/2)], fault_location_y, s=100, c='red', label='$F_{%s}$ - fault injected' %act_faulty_test)
        plt.xlabel("Time [s]")
        plt.ylabel("Surge speed [m/s]")
        plt.legend(loc='lower right')
        plt.grid()
        plt.savefig(final_dir_ + title_fig_ref, dpi=dpi_)
        plt.close(fig)
        

        title_fig_ref = "v3_comp_Reference_dynamics_x1_bounds_v2.png"
        fig = plt.figure()
        plt.plot(x_axis_scale, min_x1*0 + des_x1, '--g', label='$x_1^\star$')  # reference
        plt.fill_between(x_axis_scale, min_x1, 
                         max_x1, facecolor='blue', alpha=0.5, interpolate=True, label=c1)
        
        plt.fill_between(x_axis_scale, min_x1_lqr, 
                         max_x1_lqr, facecolor='purple', alpha=0.5, interpolate=True, label=c6)
        
        plt.fill_between(x_axis_scale, min_x1*0 + des_x1+gamma_underbar, 
                         min_x1*0 + des_x1-gamma_underbar, facecolor='green', alpha=0.2, interpolate=True, label='$\epsilon$-stability bound')
        
        fault_location_y = (max_x1[int(samples_number/2)] - min_x1[int(samples_number/2)])/2 + min_x1[int(samples_number/2)]
        plt.scatter(x_axis_scale[int(samples_number/2)], fault_location_y, s=100, c='red', label='$F_{%s}$ - fault injected' %act_faulty_test)
        fault_location_y = (max_x1_lqr[int(samples_number/2)] - min_x1_lqr[int(samples_number/2)])/2 + min_x1_lqr[int(samples_number/2)]
        plt.scatter(x_axis_scale[int(samples_number/2)], fault_location_y, s=100, c='red')
        plt.xlabel("Time [s]")
        plt.ylabel("Surge speed [m/s]")
        plt.legend(loc='lower right')
        plt.grid()
        plt.savefig(final_dir_ + title_fig_ref, dpi=dpi_)
        plt.close(fig)
        
        
        title_fig_ref = "v3_" + str(c6) + "_dynamics_x1_bounds_v2.png"
        fig, ax = plt.subplots()
        plt.plot(x_axis_scale, min_x1*0 + des_x1, '--g', label='$x_1^\star$')  # reference

        plt.fill_between(x_axis_scale, min_x1_lqr, 
                         max_x1_lqr, facecolor='purple', alpha=0.5, interpolate=True, label=c6)

        plt.fill_between(x_axis_scale, min_x2*0 + des_x1+gamma_underbar, 
                         min_x1*0 + des_x1-gamma_underbar, facecolor='green', alpha=0.2, interpolate=True, label='$\epsilon$-stability bound')

        fault_location_y = (max_x1_lqr[int(samples_number/2)] - min_x1_lqr[int(samples_number/2)])/2 + min_x1_lqr[int(samples_number/2)]
        plt.scatter(x_axis_scale[int(samples_number/2)], fault_location_y, s=100, c='red', label='$F_{%s}$ - fault injected' %act_faulty_test)
        ax.yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
        plt.xlabel("Time [s]")
        plt.ylabel("Surge speed [m/s]")
        plt.legend(loc='lower right')
        plt.grid()
        plt.savefig(final_dir_ + title_fig_ref, dpi=dpi_)
        plt.close(fig)
        

        # Dynamics 2
        title_fig_ref = "Reference_dynamics_x2_bounds_v2.png"
        fig, ax = plt.subplots()
        plt.plot(x_axis_scale, min_x2*0 + des_x2, '--g', label='$x_2^\star$')  # reference
        fault_location_y = (max_x2[int(samples_number/2)] - min_x2[int(samples_number/2)])/2 + min_x2[int(samples_number/2)]
        plt.fill_between(x_axis_scale, min_x2, 
                         max_x2, facecolor='blue', alpha=0.5, interpolate=True, label=c1)
        plt.fill_between(x_axis_scale, min_x2*0 + des_x2+gamma_underbar, 
                         min_x2*0 + des_x2-gamma_underbar, facecolor='green', alpha=0.2, interpolate=True, label='$\epsilon$-stability bound')
        plt.scatter(x_axis_scale[int(samples_number/2)], fault_location_y, s=100, c='red', label='$F_{%s}$ - fault injected' %act_faulty_test)
        ax.yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
        plt.xlabel("Time [s]")
        plt.ylabel("Angular speed [rad/s]")
        plt.legend(loc='best')
        plt.grid()
        plt.savefig(final_dir_ + title_fig_ref, dpi=dpi_)
        plt.close(fig)


        title_fig_ref = "v3_comp_Reference_dynamics_x2_bounds_v2.png"
        fig, ax = plt.subplots()
        plt.plot(x_axis_scale, min_x2*0 + des_x2, '--g', label='$x_2^\star$')  # reference
        plt.fill_between(x_axis_scale, min_x2, 
                         max_x2, facecolor='blue', alpha=0.5, interpolate=True, label=c1)
        
        plt.fill_between(x_axis_scale, min_x2_lqr, 
                         max_x2_lqr, facecolor='purple', alpha=0.5, interpolate=True, label=c6)
        
        plt.fill_between(x_axis_scale, min_x2*0 + des_x2+gamma_underbar, 
                         min_x2*0 + des_x2-gamma_underbar, facecolor='green', alpha=0.2, interpolate=True, label='$\epsilon$-stability bound')
        
        fault_location_y = (max_x2[int(samples_number/2)] - min_x2[int(samples_number/2)])/2 + min_x2[int(samples_number/2)]
        plt.scatter(x_axis_scale[int(samples_number/2)], fault_location_y, s=100, c='red', label='$F_{%s}$ - fault injected' %act_faulty_test)
        
        fault_location_y = (max_x2_lqr[int(samples_number/2)] - min_x2_lqr[int(samples_number/2)])/2 + min_x2_lqr[int(samples_number/2)]
        plt.scatter(x_axis_scale[int(samples_number/2)], fault_location_y, s=100, c='red')
        ax.yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
        plt.xlabel("Time [s]")
        plt.ylabel("Angular speed [rad/s]")
        plt.legend(loc='best')
        plt.grid()
        plt.savefig(final_dir_ + title_fig_ref, dpi=dpi_)
        plt.close(fig)


        title_fig_ref = "v3_Reference_dynamics_x2_bounds.png"
        fig, ax = plt.subplots()
        plt.plot(x_axis_scale, min_x2*0 + des_x2, '--g', label='$x_2^\star$')  # reference
        fault_location_y = (max_x2[int(samples_number/2)] - min_x2[int(samples_number/2)])/2 + min_x2[int(samples_number/2)]
        plt.fill_between(x_axis_scale, min_x2, 
                         max_x2, facecolor='blue', alpha=0.5, interpolate=True, label=c1)
        plt.fill_between(x_axis_scale, min_x2*0 + des_x2+gamma_underbar, 
                         min_x2*0 + des_x2-gamma_underbar, facecolor='green', alpha=0.2, interpolate=True, label='$\epsilon$-stability bound')
        plt.scatter(x_axis_scale[int(samples_number/2)], fault_location_y, s=100, c='red', label='$F_{%s}$ - fault injected' %act_faulty_test)
        ax.yaxis.set_major_formatter(FormatStrFormatter('%.3f'))
        plt.xlabel("Time [s]")
        plt.ylabel("Angular speed [rad/s]")
        plt.legend(loc='best')
        plt.grid()
        plt.savefig(final_dir_ + title_fig_ref, dpi=dpi_)
        plt.close(fig)


        title_fig_ref = "v3_" + str(c6) + "_dynamics_x2_bounds_v2.png"
        fig, ax = plt.subplots()
        plt.plot(x_axis_scale, min_x2*0 + des_x2, '--g', label='$x_2^\star$')  # reference

        plt.fill_between(x_axis_scale, min_x2_lqr, 
                         max_x2_lqr, facecolor='purple', alpha=0.5, interpolate=True, label=c6)

        plt.fill_between(x_axis_scale, min_x2*0 + des_x2+gamma_underbar, 
                         min_x2*0 + des_x2-gamma_underbar, facecolor='green', alpha=0.2, interpolate=True, label='$\epsilon$-stability bound')

        fault_location_y = (max_x2_lqr[int(samples_number/2)] - min_x2_lqr[int(samples_number/2)])/2 + min_x2_lqr[int(samples_number/2)]
        plt.scatter(x_axis_scale[int(samples_number/2)], fault_location_y, s=100, c='red', label='$F_{%s}$ - fault injected' %act_faulty_test)
        ax.yaxis.set_major_formatter(FormatStrFormatter('%.3f'))
        plt.xlabel("Time [s]")
        plt.ylabel("Angular speed [rad/s]")
        plt.legend(loc='best')
        plt.grid()
        plt.savefig(final_dir_ + title_fig_ref, dpi=dpi_)
        plt.close(fig)     
    

def plot_comparison_three_single(x_all, u_all, V_all, x_axis_scale, 
                    dynamic_fault_test, c1, c6, x_single, u_single, c7, act_faulty_test,
                    des_x1, des_x2, init_x1, init_x2, gamma_underbar, 
                    plot_saturation,  saturation_value,
                    samples_number, ann_files, 
                    x_all_lqr, u_all_lqr,
                    dpi_, final_dir_):  # run_statistics
    
    # This function plots one pFT-ANLC controller,  
    # one state-space controller, and one state-space controller
    
    
    # Obtaining boundaries pFT-ANLC
    x1_pftanlc = x_all[0][:,0].copy()
    x2_pftanlc = x_all[0][:,1].copy()
    u1_pftanlc = u_all[0][:,0].copy()
    u2_pftanlc = u_all[0][:,1].copy()
    u3_pftanlc = u_all[0][:,2].copy()
    V_pftanlc = V_all[0][:,0].copy()
      
    
    # Obtaining boundaries controller state-space
    x1_single_K1 = x_all_lqr[0][:,0].copy()
    x2_single_K1 = x_all_lqr[0][:,1].copy()
    u1_single_K1 = u_all_lqr[0][:,0].copy()
    u2_single_K1 = u_all_lqr[0][:,1].copy()
    u3_single_K1 = u_all_lqr[0][:,2].copy()
      

    # Obtaining controller state-space -- single tuning
    x1_single = x_single[0][:,0].copy()
    x2_single = x_single[0][:,1].copy()
    u1_single = u_single[0][:,0].copy()
    u2_single = u_single[0][:,1].copy()
    u3_single = u_single[0][:,2].copy() 


    # FORCE 1
    title_fig_c = "comp_Control_input_forces_bounds_F1.png"
    fig = plt.figure()
    plt.plot(x_axis_scale, u1_pftanlc, 'blue', linewidth=2.5, alpha=0.7, label=c1)  # single controller K1
    plt.plot(x_axis_scale, u1_single_K1, 'purple', linewidth=2.5, alpha=0.7, label=c6)  # single controller K1
    plt.plot(x_axis_scale, u1_single, 'orange', linewidth=2.5, alpha=0.7, label=c7)  # single controller
    plt.legend(loc='best')
    plt.xlabel("Time [s]")
    plt.ylabel("Force [N]")
    plt.grid()
    plt.savefig(final_dir_ + title_fig_c, dpi=dpi_)
    plt.close(fig)

    # FORCE 1 -- log scale
    title_fig_c = "comp_Control_input_forces_bounds_F1_log.png"
    fig = plt.figure()
    plt.plot(x_axis_scale, u1_pftanlc, 'blue', linewidth=2.5, alpha=0.7, label=c1)  # single controller K1
    plt.plot(x_axis_scale, u1_single_K1, 'purple', linewidth=2.5, alpha=0.7, label=c6)  # single controller K1
    plt.plot(x_axis_scale, u1_single, 'orange', linewidth=2.5, alpha=0.7, label=c7)  # single controller
    if plot_saturation:
        plt.plot(x_axis_scale, u1_pftanlc*0 + saturation_value, 
                 '--r', linewidth=1.5, alpha=0.7, label="control saturation")  # single controller
    plt.legend(loc='best')
    plt.xlabel("Time [s]")
    plt.ylabel("Force [N]")
    plt.yscale("log")
    plt.grid()
    plt.savefig(final_dir_ + title_fig_c, dpi=dpi_)
    plt.close(fig)

    # FORCE 2
    title_fig_c = "comp_Control_input_forces_bounds_F2.png"
    fig = plt.figure()
    plt.plot(x_axis_scale, u2_pftanlc, 'blue', linewidth=2.5, alpha=0.7, label=c1)  # single controller K1
    plt.plot(x_axis_scale, u2_single_K1, 'purple', linewidth=2.5, alpha=0.7, label=c6)  # single controller K1
    plt.plot(x_axis_scale, u2_single, 'orange', linewidth=2.5, alpha=0.7, label=c7)  # single controller
    plt.legend(loc='best')
    plt.xlabel("Time [s]")
    plt.ylabel("Force [N]")
    plt.grid()
    plt.savefig(final_dir_ + title_fig_c, dpi=dpi_)
    plt.close(fig)

    # FORCE 2 -- log scale
    title_fig_c = "comp_Control_input_forces_bounds_F2_log.png"
    fig = plt.figure()
    plt.plot(x_axis_scale, u2_pftanlc, 'blue', linewidth=2.5, alpha=0.7, label=c1)  # single controller K1
    plt.plot(x_axis_scale, u2_single_K1, 'purple', linewidth=2.5, alpha=0.7, label=c6)  # single controller K1
    plt.plot(x_axis_scale, u2_single, 'orange', linewidth=2.5, alpha=0.7, label=c7)  # single controller
    if plot_saturation:
        plt.plot(x_axis_scale, u2_pftanlc*0 + saturation_value, 
                 '--r', linewidth=2.5, alpha=0.7, label="control saturation")  # single controller
    plt.legend(loc='best')
    plt.xlabel("Time [s]")
    plt.ylabel("Force [N]")
    plt.yscale("log")
    plt.grid()
    plt.savefig(final_dir_ + title_fig_c, dpi=dpi_)
    plt.close(fig)


    # FORCE 3
    title_fig_c = "comp_Control_input_forces_bounds_F3.png"
    fig = plt.figure()
    plt.plot(x_axis_scale, u3_pftanlc, 'blue', linewidth=2.5, alpha=0.7, label=c1)  # single controller K1
    plt.plot(x_axis_scale, u3_single_K1, 'purple', linewidth=2.5, alpha=0.7, label=c6)  # single controller K1
    plt.plot(x_axis_scale, u3_single, 'orange', linewidth=2.5, alpha=0.7, label=c7)  # single controller
    plt.legend(loc='best')
    plt.xlabel("Time [s]")
    plt.ylabel("Force [N]")
    plt.grid()
    plt.savefig(final_dir_ + title_fig_c, dpi=dpi_)
    plt.close(fig)

    # FORCE 3 -- log scale
    title_fig_c = "comp_Control_input_forces_bounds_F3_log.png"
    fig = plt.figure()

    plt.plot(x_axis_scale, u3_pftanlc, 'blue', linewidth=2.5, alpha=0.7, label=c1)  # single controller K1
    plt.plot(x_axis_scale, u3_single_K1, 'purple', linewidth=2.5, alpha=0.7, label=c6)  # single controller K1
    plt.plot(x_axis_scale, u3_single, 'orange', linewidth=2.5, alpha=0.7, label=c7)  # single controller
    if plot_saturation:
        plt.plot(x_axis_scale, u3_pftanlc*0 + saturation_value, 
                 '--r', linewidth=2.5, alpha=0.7, label="control saturation")  # single controller
    plt.legend(loc='best')
    plt.xlabel("Time [s]")
    plt.ylabel("Force [N]")
    plt.yscale("log")
    plt.grid()
    plt.savefig(final_dir_ + title_fig_c, dpi=dpi_)
    plt.close(fig)

    if dynamic_fault_test:

        title_fig_ref = "comp_Reference_dynamics_x1_bounds_short.png"
        fig = plt.figure()
        plt.plot(x_axis_scale[int(samples_number/2-0.1*samples_number):], x1_pftanlc[int(samples_number/2-0.1*samples_number):]*0 + des_x1, '--g', label='$x_1^\star$')  # reference
        plt.plot(x_axis_scale[int(samples_number/2-0.1*samples_number):], x1_pftanlc[int(samples_number/2-0.1*samples_number):], 
                 'blue', linewidth=2.5, alpha=0.7, label=c1) 
        
        plt.plot(x_axis_scale[int(samples_number/2-0.1*samples_number):], x1_single_K1[int(samples_number/2-0.1*samples_number):], 
                 'purple', linewidth=2.5, alpha=0.7, label=c6)  # single controller K1
        
        plt.plot(x_axis_scale[int(samples_number/2-0.1*samples_number):], x1_single[int(samples_number/2-0.1*samples_number):], 
                 'orange', linewidth=2.5, alpha=0.7, label=c7)  # single controller

        plt.fill_between(x_axis_scale[int(samples_number/2-0.1*samples_number):], x1_pftanlc[int(samples_number/2-0.1*samples_number):]*0 + des_x1+gamma_underbar, 
                         x1_pftanlc[int(samples_number/2-0.1*samples_number):]*0 + des_x1-gamma_underbar, facecolor='green', alpha=0.2, interpolate=True, label='$\epsilon$-stability bound')
        plt.xlabel("Time [s]")
        plt.ylabel("Surge speed [m/s]")
        plt.legend(loc='best')
        plt.grid()
        plt.savefig(final_dir_ + title_fig_ref, dpi=dpi_)
        plt.close(fig)
              

        title_fig_ref = "v3_comp_Reference_dynamics_x1_bounds_v2.png"
        fig = plt.figure()
        plt.plot(x_axis_scale, x1_pftanlc*0 + des_x1, '--g', label='$x_1^\star$')  # reference
        plt.plot(x_axis_scale, x1_pftanlc, 
                 'blue', linewidth=2.5, alpha=0.7, label=c1) 
        plt.plot(x_axis_scale, x1_single_K1, 'purple', linewidth=2.5, alpha=0.7, label=c6)  # single controller K1

        plt.plot(x_axis_scale, x1_single, 'orange', linewidth=2.5, alpha=0.7, label=c7)  # single controller

        plt.fill_between(x_axis_scale, x1_pftanlc*0 + des_x1+gamma_underbar, 
                         x1_pftanlc*0 + des_x1-gamma_underbar, facecolor='green', alpha=0.2, interpolate=True, label='$\epsilon$-stability bound')
        
        fault_location_y = x1_pftanlc[int(samples_number/2)]
        plt.scatter(x_axis_scale[int(samples_number/2)], fault_location_y, s=100, c='red', label='$F_{%s}$ - fault injected' %act_faulty_test)
        plt.xlabel("Time [s]")
        plt.ylabel("Surge speed [m/s]")
        plt.legend(loc='lower right')
        plt.grid()
        plt.savefig(final_dir_ + title_fig_ref, dpi=dpi_)
        plt.close(fig)       
        
        
        # Dynamics 2
        title_fig_ref = "v3_comp_Reference_dynamics_x2_bounds_v2.png"
        fig, ax = plt.subplots()
        plt.plot(x_axis_scale, x2_pftanlc*0 + des_x2, '--g', label='$x_2^\star$')  # reference
        plt.plot(x_axis_scale, x2_pftanlc, 'blue', linewidth=2.5, alpha=0.7, label=c1)
        plt.plot(x_axis_scale, x2_single_K1, 'purple', linewidth=2.5, alpha=0.7, label=c6)  # single controller K1
        plt.plot(x_axis_scale, x2_single, 'orange', linewidth=2.5, alpha=0.7, label=c7)  # single controller
        plt.fill_between(x_axis_scale, x2_pftanlc*0 + des_x2+gamma_underbar, 
                         x2_pftanlc*0 + des_x2-gamma_underbar, facecolor='green', alpha=0.2, interpolate=True, label='$\epsilon$-stability bound')
        
        fault_location_y = x2_pftanlc[int(samples_number/2)]
        plt.scatter(x_axis_scale[int(samples_number/2)], fault_location_y, s=100, c='red', label='$F_{%s}$ - fault injected' %act_faulty_test)
        ax.yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
        plt.xlabel("Time [s]")
        plt.ylabel("Angular speed [rad/s]")
        plt.legend(loc='best')
        plt.grid()
        plt.savefig(final_dir_ + title_fig_ref, dpi=dpi_)
        plt.close(fig)
        


def plot_comparison_best_c(x_all, u_all, V_all, x_axis_scale, 
                           x_best, u_best, V_best,
                           dynamic_fault_test, c1, c6, act_faulty_test,
                           des_x1, des_x2, init_x1, init_x2, gamma_underbar, 
                           samples_number, ann_files, 
                           x_all_lqr, u_all_lqr,
                           dpi_, final_dir_comp, run_statistics):
    
    # This function plots the bounds of the pFT-ANLC controllers and the bounds of the 
    # state-space controller, alotogether with the 'best' selected controller
    
    
    # Obtaining boundaries pFT-ANLC
    min_x1 = x_all[0][:,0].copy()
    min_x2 = x_all[0][:,1].copy()
    min_u1 = u_all[0][:,0].copy()
    min_u2 = u_all[0][:,1].copy()
    min_u3 = u_all[0][:,2].copy()
    min_V = V_all[0][:,0].copy()
   
    max_x1 = x_all[0][:,0].copy()
    max_x2 = x_all[0][:,1].copy()
    max_u1 = u_all[0][:,0].copy()
    max_u2 = u_all[0][:,1].copy()
    max_u3 = u_all[0][:,2].copy()
    max_V = V_all[0][:,0].copy()
    
    for jCntr in range(ann_files):
        min_x1 = np.minimum(min_x1, x_all[jCntr][:,0]).copy()
        min_x2 = np.minimum(min_x2, x_all[jCntr][:,1]).copy()
        min_u1 = np.minimum(min_u1, u_all[jCntr][:,0]).copy()
        min_u2 = np.minimum(min_u2, u_all[jCntr][:,1]).copy()
        min_u3 = np.minimum(min_u3, u_all[jCntr][:,2]).copy()
        min_V = np.minimum(min_V, V_all[jCntr][:,0]).copy()

        max_x1 = np.maximum(max_x1, x_all[jCntr][:,0]).copy()
        max_x2 = np.maximum(max_x2, x_all[jCntr][:,1]).copy()
        max_u1 = np.maximum(max_u1, u_all[jCntr][:,0]).copy()
        max_u2 = np.maximum(max_u2, u_all[jCntr][:,1]).copy()
        max_u3 = np.maximum(max_u3, u_all[jCntr][:,2]).copy()
        max_V = np.maximum(max_V, V_all[jCntr][:,0]).copy()
    
    
    
    # Obtaining boundaries controller state-space
    min_x1_lqr = x_all_lqr[0][:,0].copy()
    min_x2_lqr = x_all_lqr[0][:,1].copy()
    min_u1_lqr = u_all_lqr[0][:,0].copy()
    min_u2_lqr = u_all_lqr[0][:,1].copy()
    min_u3_lqr = u_all_lqr[0][:,2].copy()
   
    max_x1_lqr = x_all_lqr[0][:,0].copy()
    max_x2_lqr = x_all_lqr[0][:,1].copy()
    max_u1_lqr = u_all_lqr[0][:,0].copy()
    max_u2_lqr = u_all_lqr[0][:,1].copy()
    max_u3_lqr = u_all_lqr[0][:,2].copy()
    

    for jCntr in range(2):
        min_x1_lqr = np.minimum(min_x1_lqr, x_all_lqr[jCntr][:,0]).copy()
        min_x2_lqr = np.minimum(min_x2_lqr, x_all_lqr[jCntr][:,1]).copy()
        min_u1_lqr = np.minimum(min_u1_lqr, u_all_lqr[jCntr][:,0]).copy()
        min_u2_lqr = np.minimum(min_u2_lqr, u_all_lqr[jCntr][:,1]).copy()
        min_u3_lqr = np.minimum(min_u3_lqr, u_all_lqr[jCntr][:,2]).copy()

        max_x1_lqr = np.maximum(max_x1_lqr, x_all_lqr[jCntr][:,0]).copy()
        max_x2_lqr = np.maximum(max_x2_lqr, x_all_lqr[jCntr][:,1]).copy()
        max_u1_lqr = np.maximum(max_u1_lqr, u_all_lqr[jCntr][:,0]).copy()
        max_u2_lqr = np.maximum(max_u2_lqr, u_all_lqr[jCntr][:,1]).copy()
        max_u3_lqr = np.maximum(max_u3_lqr, u_all_lqr[jCntr][:,2]).copy()
    
    
    # 1.1 - all forces together
    title_fig_c = "Control_input_forces_bounds_" + str(c1) +".png"
    fig = plt.figure()
    plt.fill_between(x_axis_scale, min_u1, max_u1, facecolor='blue', alpha=0.8, interpolate=True, label='$F_1$')
    plt.plot(x_axis_scale, u_best[:,0], ls=':', color='blue', label='$\overline{F}_{1}$')
    plt.fill_between(x_axis_scale, min_u2, max_u2, facecolor='red', alpha=0.8, interpolate=True, label='$F_2$')
    plt.plot(x_axis_scale, u_best[:,1], ls=':', color='red', label='$\overline{F}_{2}$')
    plt.fill_between(x_axis_scale, min_u3, max_u3, facecolor='green', alpha=0.8, interpolate=True, label='$F_3$')
    plt.plot(x_axis_scale, u_best[:,2], ls=':', color='green', label='$\overline{F}_{3}$')
    plt.legend(loc='best')
    plt.xlabel("Time [s]")
    plt.ylabel("Force [N]")
    plt.grid()
    plt.savefig(final_dir_comp + title_fig_c, dpi=dpi_)
    plt.close(fig)
    
    
    if dynamic_fault_test:
        # 1.1 - all forces together
        title_fig_c = "Control_input_forces_bounds_v2_" + str(c1) +".png"
        fig = plt.figure()
        
        plt.fill_between(x_axis_scale, min_u1, max_u1, facecolor='blue', alpha=0.8, interpolate=True, label='$F_1$')
        plt.plot(x_axis_scale, u_best[:,0], ls=':', color='blue', label='$\overline{F}_{1}$')
        plt.fill_between(x_axis_scale, min_u2, max_u2, facecolor='red', alpha=0.8, interpolate=True, label='$F_2$')
        plt.plot(x_axis_scale, u_best[:,1], ls=':', color='red', label='$\overline{F}_{2}$')
        plt.fill_between(x_axis_scale, min_u3, max_u3, facecolor='green', alpha=0.8, interpolate=True, label='$F_3$')
        plt.plot(x_axis_scale, u_best[:,2], ls=':', color='green', label='$\overline{F}_{3}$')
        
        if act_faulty_test==1:
            fault_location_y = (max_u1[int(samples_number/2)] - min_u1[int(samples_number/2)])/2 + min_u1[int(samples_number/2)]
        if act_faulty_test==2:
            fault_location_y = (max_u2[int(samples_number/2)] - min_u2[int(samples_number/2)])/2 + min_u2[int(samples_number/2)]
        if act_faulty_test==3:
            fault_location_y = (max_u3[int(samples_number/2)] - min_u3[int(samples_number/2)])/2 + min_u3[int(samples_number/2)]
        
        plt.scatter(x_axis_scale[int(samples_number/2)], fault_location_y, s=100, c='red', label='$F_{%s}$ - fault injected' %act_faulty_test)
        plt.legend(loc='best')
        plt.xlabel("Time [s]")
        plt.ylabel("Force [N]")
        plt.grid()
        plt.savefig(final_dir_comp + title_fig_c, dpi=dpi_)
        plt.close(fig)
    
    
    # 1.1 - all forces together - half time 
    t_short = 2
    title_fig_c = "Control_input_forces_bounds_t" + str(t_short) + "_" + str(c1) +".png"
    fig = plt.figure()
    plt.fill_between(x_axis_scale[:int(samples_number/t_short)], min_u1[:int(samples_number/t_short)], max_u1[:int(samples_number/t_short)], facecolor='blue', alpha=0.8, interpolate=True, label='$F_1$')
    plt.plot(x_axis_scale[:int(samples_number/t_short)], u_best[:int(samples_number/t_short),0], ls=':', color='blue', label='$\overline{F}_{1}$')

    plt.fill_between(x_axis_scale[:int(samples_number/t_short)], min_u2[:int(samples_number/t_short)], max_u2[:int(samples_number/t_short)], facecolor='red', alpha=0.8, interpolate=True, label='$F_2$')
    plt.plot(x_axis_scale[:int(samples_number/t_short)], u_best[:int(samples_number/t_short),1], ls=':', color='red', label='$\overline{F}_{2}$')

    plt.fill_between(x_axis_scale[:int(samples_number/t_short)], min_u3[:int(samples_number/t_short)], max_u3[:int(samples_number/t_short)], facecolor='green', alpha=0.8, interpolate=True, label='$F_3$')
    plt.plot(x_axis_scale[:int(samples_number/t_short)], u_best[:int(samples_number/t_short),2], ls=':', color='green', label='$\overline{F}_{3}$')

    plt.legend(loc='best')
    plt.xlabel("Time [s]")
    plt.ylabel("Force [N]")
    plt.grid()
    plt.savefig(final_dir_comp + title_fig_c, dpi=dpi_)
    plt.close(fig)
    

    t_short = 6
    title_fig_c = "Control_input_forces_bounds_t" + str(t_short) + "_" + str(c1) +".png"
    fig = plt.figure()
    plt.fill_between(x_axis_scale[:int(samples_number/t_short)], min_u1[:int(samples_number/t_short)], max_u1[:int(samples_number/t_short)], facecolor='blue', alpha=0.8, interpolate=True, label='$F_1$')
    plt.plot(x_axis_scale[:int(samples_number/t_short)], u_best[:int(samples_number/t_short),0], ls=':', color='blue', label='$\overline{F}_{1}$')
    
    plt.fill_between(x_axis_scale[:int(samples_number/t_short)], min_u2[:int(samples_number/t_short)], max_u2[:int(samples_number/t_short)], facecolor='red', alpha=0.8, interpolate=True, label='$F_2$')
    plt.plot(x_axis_scale[:int(samples_number/t_short)], u_best[:int(samples_number/t_short),1], ls=':', color='red', label='$\overline{F}_{2}$')

    plt.fill_between(x_axis_scale[:int(samples_number/t_short)], min_u3[:int(samples_number/t_short)], max_u3[:int(samples_number/t_short)], facecolor='green', alpha=0.8, interpolate=True, label='$F_3$')
    plt.plot(x_axis_scale[:int(samples_number/t_short)], u_best[:int(samples_number/t_short),2], ls=':', color='green', label='$\overline{F}_{3}$')

    plt.legend(loc='best')
    plt.xlabel("Time [s]")
    plt.ylabel("Force [N]")
    plt.grid()
    plt.savefig(final_dir_comp + title_fig_c, dpi=dpi_)
    plt.close(fig)


    if dynamic_fault_test:
        # 1.1 - all forces together (final section)
        title_fig_c = "Control_input_forces_short_bounds_" + str(c1) +".png"
        fig = plt.figure()
        plt.fill_between(x_axis_scale[int(samples_number/2-0.1*samples_number):], min_u1[int(samples_number/2-0.1*samples_number):], max_u1[int(samples_number/2-0.1*samples_number):], facecolor='blue', alpha=0.8, interpolate=True, label='$F_1$')
        plt.fill_between(x_axis_scale[int(samples_number/2-0.1*samples_number):], min_u2[int(samples_number/2-0.1*samples_number):], max_u2[int(samples_number/2-0.1*samples_number):], facecolor='red', alpha=0.8, interpolate=True, label='$F_2$')
        plt.fill_between(x_axis_scale[int(samples_number/2-0.1*samples_number):], min_u3[int(samples_number/2-0.1*samples_number):], max_u3[int(samples_number/2-0.1*samples_number):], facecolor='green', alpha=0.8, interpolate=True, label='$F_3$')
        plt.legend(loc='best')
        plt.xlabel("Time [s]")
        plt.ylabel("Force [N]")
        plt.grid()
        plt.savefig(final_dir_comp + title_fig_c, dpi=dpi_)
        plt.close(fig)


        
    title_fig_ref = "Reference_dynamics_x1_bounds_" + str(c1) +".png"
    fig = plt.figure()
    plt.plot(x_axis_scale, min_x1*0 + des_x1, '--g', label='$x_1^\star$')  # reference
    
    plt.fill_between(x_axis_scale, min_x1, 
                     max_x1, facecolor='blue', alpha=0.5, interpolate=True, label=c1)
    plt.plot(x_axis_scale, x_best[:,0], ls=':', color='blue', label='$\overline{x}_{1}$')
    
    plt.fill_between(x_axis_scale, min_x1*0 + des_x1+gamma_underbar, 
                     min_x1*0 + des_x1-gamma_underbar, facecolor='green', alpha=0.2, interpolate=True, label='$\epsilon$-stability bound')
    plt.xlabel("Time [s]")
    plt.ylabel("Surge speed [m/s]")
    plt.legend(loc='best')
    plt.grid()
    plt.savefig(final_dir_comp + title_fig_ref, dpi=dpi_)
    plt.close(fig)


    t_short = 2
    title_fig_ref = "Reference_dynamics_x1_bounds_t" + str(t_short) + "_" + str(c1) +".png"
    fig = plt.figure()
    plt.plot(x_axis_scale[:int(samples_number/t_short)], min_x1[:int(samples_number/t_short)]*0 + des_x1, '--g', label='$x_1^\star$')  # reference
    plt.fill_between(x_axis_scale[:int(samples_number/t_short)], min_x1[:int(samples_number/t_short)], 
                     max_x1[:int(samples_number/t_short)], facecolor='blue', alpha=0.5, interpolate=True, label=c1)
    plt.plot(x_axis_scale[:int(samples_number/t_short)], x_best[:int(samples_number/t_short),0], ls=':', color='blue', label='$\overline{x}_{1}$')

    plt.fill_between(x_axis_scale[:int(samples_number/t_short)], min_x1[:int(samples_number/t_short)]*0 + des_x1+gamma_underbar, 
                     min_x1[:int(samples_number/t_short)]*0 + des_x1-gamma_underbar, facecolor='green', alpha=0.2, interpolate=True, label='$\epsilon$-stability bound')
    plt.xlabel("Time [s]")
    plt.ylabel("Surge speed [m/s]")
    plt.legend(loc='best')
    plt.grid()
    plt.savefig(final_dir_comp + title_fig_ref, dpi=dpi_)
    plt.close(fig)


    t_short = 6
    title_fig_ref = "Reference_dynamics_x1_bounds_t" + str(t_short) + "_" + str(c1) +".png"
    fig = plt.figure()
    plt.plot(x_axis_scale[:int(samples_number/t_short)], min_x1[:int(samples_number/t_short)]*0 + des_x1, '--g', label='$x_{1*}$')  # reference
    plt.fill_between(x_axis_scale[:int(samples_number/t_short)], min_x1[:int(samples_number/t_short)], 
                     max_x1[:int(samples_number/t_short)], facecolor='blue', alpha=0.5, interpolate=True, label=c1)
    plt.plot(x_axis_scale[:int(samples_number/t_short)], x_best[:int(samples_number/t_short),0], ls=':', color='blue', label='$\overline{x}_{1}$')
    plt.fill_between(x_axis_scale[:int(samples_number/t_short)], min_x1[:int(samples_number/t_short)]*0 + des_x1+gamma_underbar, 
                     min_x1[:int(samples_number/t_short)]*0 + des_x1-gamma_underbar, facecolor='green', alpha=0.2, interpolate=True, label='$\epsilon$-stability bound')
    plt.xlabel("Time [s]")
    plt.ylabel("Surge speed [m/s]")
    plt.legend(loc='best')
    plt.grid()
    plt.savefig(final_dir_comp + title_fig_ref, dpi=dpi_)
    plt.close(fig)


    if dynamic_fault_test:
        
        title_fig_ref = "Reference_dynamics_x1_bounds_short_" + str(c1) +".png"
        fig = plt.figure()
        plt.plot(x_axis_scale[int(samples_number/2-0.1*samples_number):], min_x1[int(samples_number/2-0.1*samples_number):]*0 + des_x1, '--g', label='$x_1^\star$')  # reference
        plt.fill_between(x_axis_scale[int(samples_number/2-0.1*samples_number):], min_x1[int(samples_number/2-0.1*samples_number):], 
                         max_x1[int(samples_number/2-0.1*samples_number):], facecolor='blue', alpha=0.5, interpolate=True, label=c1)
        plt.plot(x_axis_scale[int(samples_number/2-0.1*samples_number):], 
                     x_best[int(samples_number/2-0.1*samples_number):,0], ls=':', color='blue', label='$\overline{x}_{1}$')
        plt.fill_between(x_axis_scale[int(samples_number/2-0.1*samples_number):], min_x1[int(samples_number/2-0.1*samples_number):]*0 + des_x1+gamma_underbar, 
                         min_x1[int(samples_number/2-0.1*samples_number):]*0 + des_x1-gamma_underbar, facecolor='green', alpha=0.2, interpolate=True, label='$\epsilon$-stability bound')
        plt.xlabel("Time [s]")
        plt.ylabel("Surge speed [m/s]")
        plt.legend(loc='best')
        plt.grid()
        plt.savefig(final_dir_comp + title_fig_ref, dpi=dpi_)
        plt.close(fig)


        fault_location_y = (max_x1[int(samples_number/2)] - min_x1[int(samples_number/2)])/2 + min_x1[int(samples_number/2)]
        title_fig_ref = "Reference_dynamics_x1_bounds_short_v2_" + str(c1) +".png"
        fig = plt.figure()
        plt.plot(x_axis_scale[int(samples_number/2-0.1*samples_number):], min_x1[int(samples_number/2-0.1*samples_number):]*0 + des_x1, '--g', label='$x_1^\star$')  # reference
        plt.fill_between(x_axis_scale[int(samples_number/2-0.1*samples_number):], min_x1[int(samples_number/2-0.1*samples_number):], 
                         max_x1[int(samples_number/2-0.1*samples_number):], facecolor='blue', alpha=0.5, interpolate=True, label=c1)
        plt.plot(x_axis_scale[int(samples_number/2-0.1*samples_number):], 
                     x_best[int(samples_number/2-0.1*samples_number):,0], ls=':', color='blue', label='$\overline{x}_{1}$')
        plt.fill_between(x_axis_scale[int(samples_number/2-0.1*samples_number):], min_x1[int(samples_number/2-0.1*samples_number):]*0 + des_x1+gamma_underbar, 
                         min_x1[int(samples_number/2-0.1*samples_number):]*0 + des_x1-gamma_underbar, facecolor='green', alpha=0.2, interpolate=True, label='$\epsilon$-stability bound')
        plt.scatter(x_axis_scale[int(samples_number/2)], fault_location_y, s=200, c='red', label='$F_{%s}$ - fault injected' %act_faulty_test)
        plt.xlabel("Time [s]")
        plt.ylabel("Surge speed [m/s]")
        plt.legend(loc='best')
        plt.grid()
        plt.savefig(final_dir_comp + title_fig_ref, dpi=dpi_)
        plt.close(fig)


    # 4) Dynamics x2
    title_fig_ref = "Reference_dynamics_x2_bounds_" + str(c1) +".png"
    fig, ax = plt.subplots()
    plt.plot(x_axis_scale, min_x2*0 + des_x2, '--g', label='$x_2^\star$')  # reference
    plt.fill_between(x_axis_scale, min_x2, 
                     max_x2, facecolor='blue', alpha=0.5, interpolate=True, label=c1)
    plt.plot(x_axis_scale, x_best[:,1], ls=':', color='blue', label='$\overline{x}_{2}$')
    plt.fill_between(x_axis_scale, min_x2*0 + des_x2+gamma_underbar, 
                     min_x2*0 + des_x2-gamma_underbar, facecolor='green', alpha=0.2, interpolate=True, label='$\epsilon$-stability bound')
    ax.yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
    plt.xlabel("Time [s]")
    plt.ylabel("Angular speed [rad/s]")
    plt.legend(loc='best')
    plt.grid()
    plt.savefig(final_dir_comp + title_fig_ref, dpi=dpi_)
    plt.close(fig)


    # 4) Dynamics x2
    t_short = 2
    title_fig_ref = "Reference_dynamics_x2_bounds_t" + str(t_short) + "_" + str(c1) +".png"
    fig, ax = plt.subplots()
    plt.plot(x_axis_scale[:int(samples_number/t_short)], min_x2[:int(samples_number/t_short)]*0 + des_x2, '--g', label='$x_2^\star$')  # reference
    plt.fill_between(x_axis_scale[:int(samples_number/t_short)], min_x2[:int(samples_number/t_short)], 
                     max_x2[:int(samples_number/t_short)], facecolor='blue', alpha=0.5, interpolate=True, label=c1)
    plt.plot(x_axis_scale[:int(samples_number/t_short)], x_best[:int(samples_number/t_short),1], ls=':', color='blue', label='$\overline{x}_{1}$')
    plt.fill_between(x_axis_scale[:int(samples_number/t_short)], min_x2[:int(samples_number/t_short)]*0 + des_x2+gamma_underbar, 
                     min_x2[:int(samples_number/t_short)]*0 + des_x2-gamma_underbar, facecolor='green', alpha=0.2, interpolate=True, label='$\epsilon$-stability bound')
    ax.yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
    plt.xlabel("Time [s]")
    plt.ylabel("Angular speed [rad/s]")
    plt.legend(loc='best')
    plt.grid()
    plt.savefig(final_dir_comp + title_fig_ref, dpi=dpi_)
    plt.close(fig)


    # 4) Dynamics x2
    t_short = 6
    title_fig_ref = "Reference_dynamics_x2_bounds_t" + str(t_short) + "_" + str(c1) +".png"
    fig, ax = plt.subplots()
    plt.plot(x_axis_scale[:int(samples_number/t_short)], min_x2[:int(samples_number/t_short)]*0 + des_x2, '--g', label='$x_2^\star$')  # reference
    plt.fill_between(x_axis_scale[:int(samples_number/t_short)], min_x2[:int(samples_number/t_short)], 
                     max_x2[:int(samples_number/t_short)], facecolor='blue', alpha=0.5, interpolate=True, label=c1)
    plt.plot(x_axis_scale[:int(samples_number/t_short)], x_best[:int(samples_number/t_short),1], ls=':', color='blue', label='$\overline{x}_{1}$')
    plt.fill_between(x_axis_scale[:int(samples_number/t_short)], min_x2[:int(samples_number/t_short)]*0 + des_x2+gamma_underbar, 
                     min_x2[:int(samples_number/t_short)]*0 + des_x2-gamma_underbar, facecolor='green', alpha=0.2, interpolate=True, label='$\epsilon$-stability bound')
    ax.yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
    plt.xlabel("Time [s]")
    plt.ylabel("Angular speed [rad/s]")
    plt.legend(loc='best')
    plt.grid()
    plt.savefig(final_dir_comp + title_fig_ref, dpi=dpi_)
    plt.close(fig)


    if dynamic_fault_test:
        
        title_fig_ref = "Reference_dynamics_x2_short_bounds_"+ str(c1) +".png"
        fig, ax = plt.subplots()
        plt.plot(x_axis_scale[int(samples_number/2-0.1*samples_number):], min_x2[int(samples_number/2-0.1*samples_number):]*0 + des_x2, '--g', label='$x_2^\star$')  # reference
        plt.fill_between(x_axis_scale[int(samples_number/2-0.1*samples_number):], min_x2[int(samples_number/2-0.1*samples_number):], 
                         max_x2[int(samples_number/2-0.1*samples_number):], facecolor='blue', alpha=0.5, interpolate=True, label=c1)
        plt.plot(x_axis_scale[int(samples_number/2-0.1*samples_number):], 
                 x_best[int(samples_number/2-0.1*samples_number):,1], ls=':', color='blue', label='$\overline{x}_{1}$')
        plt.fill_between(x_axis_scale[int(samples_number/2-0.1*samples_number):], min_x2[int(samples_number/2-0.1*samples_number):]*0 + des_x2+gamma_underbar, 
                         min_x2[int(samples_number/2-0.1*samples_number):]*0 + des_x2-gamma_underbar, facecolor='green', alpha=0.2, interpolate=True, label='$\epsilon$-stability bound')
        ax.yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
        plt.xlabel("Time [s]")
        plt.ylabel("Angular speed [rad/s]")
        plt.legend(loc='best')
        plt.grid()
        plt.savefig(final_dir_comp + title_fig_ref, dpi=dpi_)
        plt.close(fig)


        fault_location_y = (max_x2[int(samples_number/2)] - min_x2[int(samples_number/2)])/2 + min_x2[int(samples_number/2)]
        title_fig_ref = "Reference_dynamics_x2_short_v2_bounds_"+ str(c1) +".png"
        fig, ax = plt.subplots()
        plt.plot(x_axis_scale[int(samples_number/2-0.1*samples_number):], min_x2[int(samples_number/2-0.1*samples_number):]*0 + des_x2, '--g', label='$x_2^\star$')  # reference
        plt.fill_between(x_axis_scale[int(samples_number/2-0.1*samples_number):], min_x2[int(samples_number/2-0.1*samples_number):], 
                         max_x2[int(samples_number/2-0.1*samples_number):], facecolor='blue', alpha=0.5, interpolate=True, label=c1)
        plt.plot(x_axis_scale[int(samples_number/2-0.1*samples_number):], 
                 x_best[int(samples_number/2-0.1*samples_number):,1], ls=':', color='blue', label='$\overline{x}_{1}$')
        plt.fill_between(x_axis_scale[int(samples_number/2-0.1*samples_number):], min_x2[int(samples_number/2-0.1*samples_number):]*0 + des_x2+gamma_underbar, 
                         min_x2[int(samples_number/2-0.1*samples_number):]*0 + des_x2-gamma_underbar, facecolor='green', alpha=0.2, interpolate=True, label='$\epsilon$-stability bound')
        plt.scatter(x_axis_scale[int(samples_number/2)], fault_location_y, s=200, c='red', label='$F_{%s}$ - fault injected' %act_faulty_test)
        ax.yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
        plt.xlabel("Time [s]")
        plt.ylabel("Angular speed [rad/s]")
        plt.legend(loc='best')
        plt.grid()
        plt.savefig(final_dir_comp + title_fig_ref, dpi=dpi_)
        plt.close(fig)


    # 3) Lyapunov value along the ANLC trajectories
    title_fig_v = "Lyapunov_value_bounds_"+ str(c1) +".png"
    fig = plt.figure()
    plt.fill_between(x_axis_scale, min_V, 
                     max_V, facecolor='blue', alpha=0.5, interpolate=True, label=c1)
    plt.plot(x_axis_scale, V_best, ls=':', color='blue', label='$\overline{V}$')
    plt.xlabel("Time [s]")
    plt.ylabel(None)
    plt.legend(loc='best')
    plt.grid()
    plt.savefig(final_dir_comp + title_fig_v, dpi=dpi_)
    plt.close(fig)
        

    # 3) Lyapunov value along the ANLC trajectories
    t_short = 2
    title_fig_v = "Lyapunov_value_bounds_t" + str(t_short) + "_" + str(c1) +".png"
    fig = plt.figure()
    plt.fill_between(x_axis_scale[:int(samples_number/t_short)], min_V[:int(samples_number/t_short)], 
                     max_V[:int(samples_number/t_short)], facecolor='blue', alpha=0.5, interpolate=True, label=c1)
    plt.plot(x_axis_scale[:int(samples_number/t_short)], 
             V_best[:int(samples_number/t_short)], ls=':', color='blue', label='$\overline{V}$')
    plt.xlabel("Time [s]")
    plt.ylabel(None)
    plt.grid()
    plt.legend(loc='best')
    plt.savefig(final_dir_comp + title_fig_v, dpi=dpi_)
    plt.close(fig)
    
    
    # 3) Lyapunov value along the ANLC trajectories
    t_short = 6
    title_fig_v = "Lyapunov_value_bounds_t" + str(t_short) + "_" + str(c1) +".png"
    fig = plt.figure()
    plt.fill_between(x_axis_scale[:int(samples_number/t_short)], min_V[:int(samples_number/t_short)], 
                     max_V[:int(samples_number/t_short)], facecolor='blue', alpha=0.5, interpolate=True, label=c1)
    plt.plot(x_axis_scale[:int(samples_number/t_short)], 
             V_best[:int(samples_number/t_short)], ls=':', color='blue', label='$\overline{V}$')
    plt.xlabel("Time [s]")
    plt.ylabel(None)
    plt.grid()
    plt.legend(loc='best')
    plt.savefig(final_dir_comp + title_fig_v, dpi=dpi_)
    plt.close(fig)


    # saving performance report
    range_x1 = max(max_x1) - min(min_x1)
    range_x2 = max(max_x2) - min(min_x2)
    ideal_x1 = des_x1 - init_x1
    ideal_x2 = des_x2 - init_x2
    
    perfo_report = [f"Max x1 = {max(max_x1)}\
                    \nMin x1 = {min(min_x1)}\
                    \nOvershoot x1 % = {(range_x1-ideal_x1)/ideal_x1*100}%\
                    \nMax x2 = {max(max_x2)}\
                    \nMin x2 = {min(min_x2)}\
                    \nOvershoot x2 % = {(range_x2-ideal_x2)/ideal_x2*100}%\
                    \nMax(abs(u1)) = {max(abs(max_u1))}\
                    \nMax(abs(u2)) = {max(abs(max_u2))}\
                    \nMax(abs(u3)) = {max(abs(max_u3))}\
                    "]
    np.savetxt(final_dir_comp + "perfo_report.txt", perfo_report, fmt="%s")



    # COMPARISON
    t_short = 1
    title_fig_ref = "comp_Reference_dynamics_x1_bounds_t" + str(t_short) + "_comp.png"
    fig = plt.figure()
    plt.plot(x_axis_scale[:int(samples_number/t_short)], min_x1[:int(samples_number/t_short)]*0 + des_x1, '--g', label='$x_1^\star$')  # reference
    
    plt.fill_between(x_axis_scale[:int(samples_number/t_short)], min_x1[:int(samples_number/t_short)], 
                     max_x1[:int(samples_number/t_short)], facecolor='blue', alpha=0.5, interpolate=True, label=c1)
    plt.plot(x_axis_scale, x_best[:,0], ls=':', color='blue', label='$\overline{x}_{1}$')
    plt.fill_between(x_axis_scale[:int(samples_number/t_short)], min_x1_lqr[:int(samples_number/t_short)], 
                     max_x1_lqr[:int(samples_number/t_short)], facecolor='purple', alpha=0.5, interpolate=True, label=c6)
    
    plt.fill_between(x_axis_scale[:int(samples_number/t_short)], min_x1[:int(samples_number/t_short)]*0 + des_x1+gamma_underbar, 
                     min_x1[:int(samples_number/t_short)]*0 + des_x1-gamma_underbar, facecolor='green', alpha=0.2, interpolate=True, label='$\epsilon$-stability bound')
    
    plt.xlabel("Time [s]")
    plt.ylabel("Surge speed [m/s]")
    plt.legend(loc='lower right')
    plt.grid()
    plt.savefig(final_dir_comp + title_fig_ref, dpi=dpi_)
    plt.close(fig)
    
    
    
    t_short = 2
    title_fig_ref = "comp_Reference_dynamics_x1_bounds_t" + str(t_short) + "_comp.png"
    fig = plt.figure()
    plt.plot(x_axis_scale[:int(samples_number/t_short)], min_x1[:int(samples_number/t_short)]*0 + des_x1, '--g', label='$x_1^\star$')  # reference
    
    plt.fill_between(x_axis_scale[:int(samples_number/t_short)], min_x1[:int(samples_number/t_short)], 
                     max_x1[:int(samples_number/t_short)], facecolor='blue', alpha=0.5, interpolate=True, label=c1)
    plt.plot(x_axis_scale[:int(samples_number/t_short)], 
             x_best[:int(samples_number/t_short),0], ls=':', color='blue', label='$\overline{x}_{1}$')
    
    plt.fill_between(x_axis_scale[:int(samples_number/t_short)], min_x1_lqr[:int(samples_number/t_short)], 
                     max_x1_lqr[:int(samples_number/t_short)], facecolor='purple', alpha=0.5, interpolate=True, label=c6)
    
    plt.fill_between(x_axis_scale[:int(samples_number/t_short)], min_x1[:int(samples_number/t_short)]*0 + des_x1+gamma_underbar, 
                     min_x1[:int(samples_number/t_short)]*0 + des_x1-gamma_underbar, facecolor='green', alpha=0.2, interpolate=True, label='$\epsilon$-stability bound')
    
    plt.xlabel("Time [s]")
    plt.ylabel("Surge speed [m/s]")
    plt.legend(loc='best')
    plt.grid()
    plt.savefig(final_dir_comp + title_fig_ref, dpi=dpi_)
    plt.close(fig)


    t_short = 4
    title_fig_ref = "comp_Reference_dynamics_x1_bounds_t" + str(t_short) + "_comp.png"
    fig = plt.figure()
    plt.plot(x_axis_scale[:int(samples_number/t_short)], min_x1[:int(samples_number/t_short)]*0 + des_x1, '--g', label='$x_1^\star$')  # reference
    
    plt.fill_between(x_axis_scale[:int(samples_number/t_short)], min_x1[:int(samples_number/t_short)], 
                     max_x1[:int(samples_number/t_short)], facecolor='blue', alpha=0.5, interpolate=True, label=c1)
    plt.plot(x_axis_scale[:int(samples_number/t_short)], 
             x_best[:int(samples_number/t_short),0], ls=':', color='blue', label='$\overline{x}_{1}$')
    plt.fill_between(x_axis_scale[:int(samples_number/t_short)], min_x1_lqr[:int(samples_number/t_short)], 
                     max_x1_lqr[:int(samples_number/t_short)], facecolor='purple', alpha=0.5, interpolate=True, label=c6)
    
    plt.fill_between(x_axis_scale[:int(samples_number/t_short)], min_x1[:int(samples_number/t_short)]*0 + des_x1+gamma_underbar, 
                     min_x1[:int(samples_number/t_short)]*0 + des_x1-gamma_underbar, facecolor='green', alpha=0.2, interpolate=True, label='$\epsilon$-stability bound')
    
    plt.xlabel("Time [s]")
    plt.ylabel("Surge speed [m/s]")
    plt.legend(loc='best')
    plt.grid()
    plt.savefig(final_dir_comp + title_fig_ref, dpi=dpi_)
    plt.close(fig)



    # FORCE 1
    title_fig_c = "comp_Control_input_forces_bounds_F1.png"
    fig = plt.figure()
    plt.fill_between(x_axis_scale, min_u1, max_u1, facecolor='red', alpha=0.8, interpolate=True, label=c1)
    plt.plot(x_axis_scale, u_best[:,0], ls=':', color='red', label='$\overline{F}_{1}$')
    plt.fill_between(x_axis_scale, min_u1_lqr, max_u1_lqr, facecolor='blue', alpha=0.8, interpolate=True, label=c6)
    plt.legend(loc='best')
    plt.xlabel("Time [s]")
    plt.ylabel("Force [N]")
    plt.grid()
    plt.savefig(final_dir_comp + title_fig_c, dpi=dpi_)
    plt.close(fig)

    # FORCE 2
    title_fig_c = "comp_Control_input_forces_bounds_F2.png"
    fig = plt.figure()
    plt.fill_between(x_axis_scale, min_u2, max_u2, facecolor='red', alpha=0.8, interpolate=True, label=c1)
    plt.plot(x_axis_scale, u_best[:,1], ls=':', color='red', label='$\overline{F}_{2}$')
    plt.fill_between(x_axis_scale, min_u2_lqr, max_u2_lqr, facecolor='blue', alpha=0.8, interpolate=True, label=c6)
    plt.legend(loc='best')
    plt.xlabel("Time [s]")
    plt.ylabel("Force [N]")
    plt.grid()
    plt.savefig(final_dir_comp + title_fig_c, dpi=dpi_)
    plt.close(fig)


    # FORCE 3
    title_fig_c = "comp_Control_input_forces_bounds_F3.png"
    fig = plt.figure()
    plt.fill_between(x_axis_scale, min_u3, max_u3, facecolor='red', alpha=0.8, interpolate=True, label=c1)
    plt.plot(x_axis_scale, u_best[:,2], ls=':', color='red', label='$\overline{F}_{3}$')
    plt.fill_between(x_axis_scale, min_u3_lqr, max_u3_lqr, facecolor='blue', alpha=0.8, interpolate=True, label=c6)
    plt.legend(loc='best')
    plt.xlabel("Time [s]")
    plt.ylabel("Force [N]")
    plt.grid()
    plt.savefig(final_dir_comp + title_fig_c, dpi=dpi_)
    plt.close(fig)


    if dynamic_fault_test:

        title_fig_ref = "comp_Reference_dynamics_x1_bounds_short.png"
        fig = plt.figure()
        plt.plot(x_axis_scale[int(samples_number/2-0.1*samples_number):], min_x1[int(samples_number/2-0.1*samples_number):]*0 + des_x1, '--g', label='$x_1^\star$')  # reference
        plt.fill_between(x_axis_scale[int(samples_number/2-0.1*samples_number):], min_x1[int(samples_number/2-0.1*samples_number):], 
                         max_x1[int(samples_number/2-0.1*samples_number):], facecolor='blue', alpha=0.5, interpolate=True, label=c1)
        plt.plot(x_axis_scale[int(samples_number/2-0.1*samples_number):], 
                 x_best[int(samples_number/2-0.1*samples_number):,0], ls=':', color='blue', label='$\overline{x}_{1}$')

        plt.fill_between(x_axis_scale[int(samples_number/2-0.1*samples_number):], min_x1_lqr[int(samples_number/2-0.1*samples_number):], 
                         max_x1_lqr[int(samples_number/2-0.1*samples_number):], facecolor='purple', alpha=0.5, interpolate=True, label=c6)        
        
        plt.fill_between(x_axis_scale[int(samples_number/2-0.1*samples_number):], min_x1[int(samples_number/2-0.1*samples_number):]*0 + des_x1+gamma_underbar, 
                         min_x1[int(samples_number/2-0.1*samples_number):]*0 + des_x1-gamma_underbar, facecolor='green', alpha=0.2, interpolate=True, label='$\epsilon$-stability bound')
        plt.xlabel("Time [s]")
        plt.ylabel("Surge speed [m/s]")
        plt.legend(loc='best')
        plt.grid()
        plt.savefig(final_dir_comp + title_fig_ref, dpi=dpi_)
        plt.close(fig)

    if dynamic_fault_test:

        # Dynamics 1
        title_fig_ref = "v3_Reference_dynamics_x1_bounds.png"
        fig = plt.figure()
        plt.plot(x_axis_scale, min_x1*0 + des_x1, '--g', label='$x_1^\star$')  # reference
        fault_location_y = (max_x1[int(samples_number/2)] - min_x1[int(samples_number/2)])/2 + min_x1[int(samples_number/2)]
        plt.fill_between(x_axis_scale, min_x1, 
                         max_x1, facecolor='blue', alpha=0.5, interpolate=True, label=c1)
        plt.plot(x_axis_scale, x_best[:,0], ls=':', color='blue', label='$\overline{x}_{1}$')
        plt.fill_between(x_axis_scale, min_x1*0 + des_x1+gamma_underbar, 
                         min_x1*0 + des_x1-gamma_underbar, facecolor='green', alpha=0.2, interpolate=True, label='$\epsilon$-stability bound')
        plt.scatter(x_axis_scale[int(samples_number/2)], fault_location_y, s=100, c='red', label='$F_{%s}$ - fault injected' %act_faulty_test)
        plt.xlabel("Time [s]")
        plt.ylabel("Surge speed [m/s]")
        plt.legend(loc='best')
        plt.grid()
        plt.savefig(final_dir_comp + title_fig_ref, dpi=dpi_)
        plt.close(fig)
        

        title_fig_ref = "v3_comp_Reference_dynamics_x1_bounds_v2.png"
        fig = plt.figure()
        plt.plot(x_axis_scale, min_x1*0 + des_x1, '--g', label='$x_1^\star$')  # reference
        plt.fill_between(x_axis_scale, min_x1, 
                         max_x1, facecolor='blue', alpha=0.5, interpolate=True, label=c1)
        plt.plot(x_axis_scale, x_best[:,0], ls=':', color='blue', label='$\overline{x}_{1}$')

        plt.fill_between(x_axis_scale, min_x1_lqr, 
                         max_x1_lqr, facecolor='purple', alpha=0.5, interpolate=True, label=c6)
        
        plt.fill_between(x_axis_scale, min_x1*0 + des_x1+gamma_underbar, 
                         min_x1*0 + des_x1-gamma_underbar, facecolor='green', alpha=0.2, interpolate=True, label='$\epsilon$-stability bound')
        
        fault_location_y = (max_x1[int(samples_number/2)] - min_x1[int(samples_number/2)])/2 + min_x1[int(samples_number/2)]
        plt.scatter(x_axis_scale[int(samples_number/2)], fault_location_y, s=100, c='red', label='$F_{%s}$ - fault injected' %act_faulty_test)
        fault_location_y = (max_x1_lqr[int(samples_number/2)] - min_x1_lqr[int(samples_number/2)])/2 + min_x1_lqr[int(samples_number/2)]
        plt.scatter(x_axis_scale[int(samples_number/2)], fault_location_y, s=100, c='red')
        plt.xlabel("Time [s]")
        plt.ylabel("Surge speed [m/s]")
        plt.legend(loc='lower right')
        plt.grid()
        plt.savefig(final_dir_comp + title_fig_ref, dpi=dpi_)
        plt.close(fig)
        
        
        title_fig_ref = "v3_" + str(c6) + "_dynamics_x1_bounds_v2.png"
        fig, ax = plt.subplots()
        plt.plot(x_axis_scale, min_x1*0 + des_x1, '--g', label='$x_1^\star$')  # reference

        plt.fill_between(x_axis_scale, min_x1_lqr, 
                         max_x1_lqr, facecolor='purple', alpha=0.5, interpolate=True, label=c6)

        plt.fill_between(x_axis_scale, min_x2*0 + des_x1+gamma_underbar, 
                         min_x1*0 + des_x2-gamma_underbar, facecolor='green', alpha=0.2, interpolate=True, label='$\epsilon$-stability bound')

        fault_location_y = (max_x1_lqr[int(samples_number/2)] - min_x1_lqr[int(samples_number/2)])/2 + min_x1_lqr[int(samples_number/2)]
        plt.scatter(x_axis_scale[int(samples_number/2)], fault_location_y, s=100, c='red', label='$F_{%s}$ - fault injected' %act_faulty_test)
        ax.yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
        plt.xlabel("Time [s]")
        plt.ylabel("Surge speed [m/s]")
        plt.legend(loc='best')
        plt.grid()
        plt.savefig(final_dir_comp + title_fig_ref, dpi=dpi_)
        plt.close(fig)
        

        # Dynamics 2
        title_fig_ref = "Reference_dynamics_x2_bounds_v2.png"
        fig, ax = plt.subplots()
        plt.plot(x_axis_scale, min_x2*0 + des_x2, '--g', label='$x_2^\star$')  # reference
        fault_location_y = (max_x2[int(samples_number/2)] - min_x2[int(samples_number/2)])/2 + min_x2[int(samples_number/2)]
        plt.fill_between(x_axis_scale, min_x2, 
                         max_x2, facecolor='blue', alpha=0.5, interpolate=True, label=c1)
        plt.plot(x_axis_scale, x_best[:,1], ls=':', color='blue', label='$\overline{x}_{1}$')
        plt.fill_between(x_axis_scale, min_x2*0 + des_x2+gamma_underbar, 
                         min_x2*0 + des_x2-gamma_underbar, facecolor='green', alpha=0.2, interpolate=True, label='$\epsilon$-stability bound')
        plt.scatter(x_axis_scale[int(samples_number/2)], fault_location_y, s=100, c='red', label='$F_{%s}$ - fault injected' %act_faulty_test)
        ax.yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
        plt.xlabel("Time [s]")
        plt.ylabel("Angular speed [rad/s]")
        plt.legend(loc='best')
        plt.grid()
        plt.savefig(final_dir_comp + title_fig_ref, dpi=dpi_)
        plt.close(fig)
        
        
        title_fig_ref = "v3_comp_Reference_dynamics_x2_bounds_v2.png"
        fig, ax = plt.subplots()
        plt.plot(x_axis_scale, min_x2*0 + des_x2, '--g', label='$x_2^\star$')  # reference
        plt.fill_between(x_axis_scale, min_x2, 
                         max_x2, facecolor='blue', alpha=0.5, interpolate=True, label=c1)
        plt.plot(x_axis_scale, x_best[:,1], ls=':', color='blue', label='$\overline{x}_{1}$')

        plt.fill_between(x_axis_scale, min_x2_lqr, 
                         max_x2_lqr, facecolor='purple', alpha=0.5, interpolate=True, label=c6)
        
        plt.fill_between(x_axis_scale, min_x2*0 + des_x2+gamma_underbar, 
                         min_x2*0 + des_x2-gamma_underbar, facecolor='green', alpha=0.2, interpolate=True, label='$\epsilon$-stability bound')
        
        fault_location_y = (max_x2[int(samples_number/2)] - min_x2[int(samples_number/2)])/2 + min_x2[int(samples_number/2)]
        plt.scatter(x_axis_scale[int(samples_number/2)], fault_location_y, s=100, c='red', label='$F_{%s}$ - fault injected' %act_faulty_test)
        
        fault_location_y = (max_x2_lqr[int(samples_number/2)] - min_x2_lqr[int(samples_number/2)])/2 + min_x2_lqr[int(samples_number/2)]
        plt.scatter(x_axis_scale[int(samples_number/2)], fault_location_y, s=100, c='red')
        ax.yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
        plt.xlabel("Time [s]")
        plt.ylabel("Angular speed [rad/s]")
        plt.legend(loc='best')
        plt.grid()
        plt.savefig(final_dir_comp + title_fig_ref, dpi=dpi_)
        plt.close(fig)
        
        
        title_fig_ref = "v3_Reference_dynamics_x2_bounds.png"
        fig, ax = plt.subplots()
        plt.plot(x_axis_scale, min_x2*0 + des_x2, '--g', label='$x_2^\star$')  # reference
        fault_location_y = (max_x2[int(samples_number/2)] - min_x2[int(samples_number/2)])/2 + min_x2[int(samples_number/2)]
        plt.fill_between(x_axis_scale, min_x2, 
                         max_x2, facecolor='blue', alpha=0.5, interpolate=True, label=c1)
        plt.plot(x_axis_scale, x_best[:,1], ls=':', color='blue', label='$\overline{x}_{1}$')
        plt.fill_between(x_axis_scale, min_x2*0 + des_x2+gamma_underbar, 
                         min_x2*0 + des_x2-gamma_underbar, facecolor='green', alpha=0.2, interpolate=True, label='$\epsilon$-stability bound')
        plt.scatter(x_axis_scale[int(samples_number/2)], fault_location_y, s=100, c='red', label='$F_{%s}$ - fault injected' %act_faulty_test)
        ax.yaxis.set_major_formatter(FormatStrFormatter('%.3f'))
        plt.xlabel("Time [s]")
        plt.ylabel("Angular speed [rad/s]")
        plt.legend(loc='best')
        plt.grid()
        plt.savefig(final_dir_comp + title_fig_ref, dpi=dpi_)
        plt.close(fig)


        title_fig_ref = "v3_" + str(c6) + "_dynamics_x2_bounds_v2.png"
        fig, ax = plt.subplots()
        plt.plot(x_axis_scale, min_x2*0 + des_x2, '--g', label='$x_2^\star$')  # reference

        plt.fill_between(x_axis_scale, min_x2_lqr, 
                         max_x2_lqr, facecolor='purple', alpha=0.5, interpolate=True, label=c6)

        plt.fill_between(x_axis_scale, min_x2*0 + des_x2+gamma_underbar, 
                         min_x2*0 + des_x2-gamma_underbar, facecolor='green', alpha=0.2, interpolate=True, label='$\epsilon$-stability bound')

        fault_location_y = (max_x2_lqr[int(samples_number/2)] - min_x2_lqr[int(samples_number/2)])/2 + min_x2_lqr[int(samples_number/2)]
        plt.scatter(x_axis_scale[int(samples_number/2)], fault_location_y, s=100, c='red', label='$F_{%s}$ - fault injected' %act_faulty_test)
        ax.yaxis.set_major_formatter(FormatStrFormatter('%.3f'))
        plt.xlabel("Time [s]")
        plt.ylabel("Angular speed [rad/s]")
        plt.legend(loc='best')
        plt.grid()
        plt.savefig(final_dir_comp + title_fig_ref, dpi=dpi_)
        plt.close(fig)
