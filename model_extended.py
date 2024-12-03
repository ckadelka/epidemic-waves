# -*- coding: utf-8 -*-
"""
Created on Fri Oct 18 22:35:20 2024

@author: shahriar, ckadelka
"""

import numpy as np
from numba import jit
import math


@jit(nopython=True)
def SEIR_with_delayed_reduction(y, t, beta, gamma, beta_e, e, c, k, N, log10_delayed_prevalence, dt, case):
    S, E, I, R = y
    if case=='Hill':
        reduction = 1 - 1 / (1 + (c/log10_delayed_prevalence)**k)
    elif case=='sigmoid':
        reduction = 1/(1 + math.exp(-k*(10**log10_delayed_prevalence - c)))
    elif case=='noReduction':
        reduction = 0
    dy = np.zeros(4,dtype=np.float64)
    newly_infected = beta * (1 - reduction) * S * (beta_e * E + I) / N
    dy[0] = -newly_infected
    dy[1] = newly_infected - e * E
    dy[2] = e * E - gamma * I
    dy[3] = gamma * I
    return dy

@jit(nopython=True)
def RK4(func, X0, ts, beta, gamma, beta_e, e, c, k, N, tau, dt, case): 
    """
    Runge Kutta 4 solver.
    """
    nt = len(ts)
    X  = np.zeros((nt, 4),dtype=np.float64)
    X[0,:] = X0
    
    assert case in ['Hill','sigmoid','noReduction'], "case needs to be 'Hill' or 'sigmoid' or 'noReduction'"
    
    delay_steps = round(tau / dt)
    log10_delayed_prevalence = math.log10(X0[2]/N)
    for i in range(nt-1):
        if i>delay_steps:
            log10_delayed_prevalence = math.log10(X[i-delay_steps][2]/N)            
        k1 = func(X[i], ts[i],beta, gamma, beta_e, e, c, k, N, log10_delayed_prevalence, dt, case)
        k2 = func(X[i] + dt*k1/2., ts[i] + dt/2.,beta, gamma, beta_e, e, c, k, N, log10_delayed_prevalence, dt, case)
        k3 = func(X[i] + dt*k2/2., ts[i] + dt/2.,beta, gamma, beta_e, e, c, k, N, log10_delayed_prevalence, dt, case)
        k4 = func(X[i] + dt*k3, ts[i] + dt,beta, gamma, beta_e, e, c, k, N, log10_delayed_prevalence, dt, case)
        X[i+1] = X[i] + dt / 6. * (k1 + 2. * k2 + 2. * k3 + k4)
    return X

def simulate(N=5000, I0_proportion=0.0002, beta=0.4, gamma=0.2, beta_e=0, e=0.2, c=2, k=16, tau=0, dt=0.1, case='Hill', t_end=500):
    ts = np.linspace(0, t_end, round(t_end / dt) + 1)
    x0 = np.array([N*(1-I0_proportion), N*I0_proportion * 1/e / (1/e + 1/gamma),N*I0_proportion * 1/gamma / (1/e + 1/gamma), 0],dtype=np.float64)
    if case=='Hill':
        c = np.log10(c/100)
    if case=='sigmoid':
        c = c/100
    results = RK4(SEIR_with_delayed_reduction, x0, ts, beta, gamma, beta_e, e, c, k, N, tau, dt, case)
    
    delay_steps = round(tau / dt)
    if delay_steps>0:
        delayed_Is = np.append(x0[2]*np.ones(delay_steps) , results[:-delay_steps,2])
    else:
        delayed_Is = results[:,2]
    if case=='Hill':
        reduction = 1 - 1 / (1 + (c/np.log10(delayed_Is/N))**k)
    elif case=='sigmoid':
        reduction = 1/(1 + np.exp(-k*(delayed_Is/N - c)))
    else:
        reduction = np.zeros(len(ts))
    Reffs = (1 - reduction) * (results[:, 0] / N) * (beta / gamma)
    return ts, results/N*100, reduction, Reffs

    
    