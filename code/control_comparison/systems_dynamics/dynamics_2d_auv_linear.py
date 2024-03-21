#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 25 15:00:07 2023

@authors: Davide Grande
          Andrea Peruffo

A script containing the discrete dynamics of linear AUV.

"""
import numpy as np
import torch

#
# AUV linear dynamics
#
def dyn(x, u, Dt, parameters):
    
    m = parameters['m']
    Jz = parameters['Jz']
    Xu = parameters['Xu']
    Nr = parameters['Nr']
    l1 = parameters['l1']
    l2 = parameters['l2']
    l3 = parameters['l3']
    alpha = parameters['alpha']
    gamma = parameters['gamma']

    h1 = parameters['h1']
    h2 = parameters['h2']
    h3 = parameters['h3']

    x1, x2 = x
    u1, u2, u3 = u

    
    dydt = [(-Xu*x1 + u1*h1*np.cos(torch.tensor(alpha))                + u2*h2*np.cos(torch.tensor(alpha)))/m * Dt + x1,
            (-Nr*x2 + u1*h1*l1*np.sin(torch.tensor(gamma)) -u2*h2*l2 * np.sin(torch.tensor(gamma)) -u3*h3*l3) /Jz * Dt + x2]


    return torch.Tensor([dydt])
