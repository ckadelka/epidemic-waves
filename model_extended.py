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
    """
    Simulates the SIER model with a delayed reduction mechanism based on a prevalence-dependent function.

    Parameters:
    ----------
    y : array-like
        Initial state vector [S, E, I, R], where:
          S : Susceptible population.
          E : Exponsed population.
          I : Infected population.
          R : Recovered population.
    t : float
        Current time step.
    beta : float
        Transmission rate.
    gamma : float
        Recovery rate, i.e. Reciprocal of the average time in infected state.
    beta_e : float
        Relative contagiousness of exposed individuals (compared to infected individuals).
    e : float
        Reciprocal of the average time in exposed state.
    c : float
        Behavioral response midpoint.
    k : float
        Behavioral response sensitivity.
    N : int
        Total population.
    log10_delayed_prevalence : float
        Logarithm (base 10) of delayed prevalence (I / N).
    dt : float
        Time step size.
    case : str
        Reduction mechanism:
          'Hill' : Hill function-based reduction.
          'sigmoid' : Sigmoid-based reduction.
          'noReduction' : No reduction in transmission.

    Returns:
    -------
    dy : array-like
        Rate of change [dS, dE, dI, dR].
    """

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
    Implements the 4th-order Runge-Kutta (RK4) method to solve the SIR model.

    Parameters:
    ----------
    func : callable
        Function to evaluate derivatives (e.g., SEIR_with_delayed_reduction).
    X0 : array-like
        Initial state vector [S, E, I, R].
    ts : array-like
        Time points for simulation.
    beta : float
        Transmission rate.
    gamma : float
        Recovery rate, i.e. Reciprocal of the average time in infected state.
    beta_e : float
        Relative contagiousness of exposed individuals (compared to infected individuals).
    e : float
        Reciprocal of the average time in exposed state.
    c : float
        Behavioral response midpoint.
    k : float
        Behavioral response sensitivity.
    N : int
        Total population.
    tau : float
        Delay time for reduction to take effect.
    dt : float
        Time step size.
    case : str
        Reduction mechanism:
          'Hill' : Hill function-based reduction.
          'sigmoid' : Sigmoid-based reduction.
          'noReduction' : No reduction in transmission.

    Returns:
    -------
    X : array-like
        Simulated trajectories of [S, E, I, R] at each time step.
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
    """
    Computes the dynamics of the SEIR model with delayed human behavioral responses and calculates additional outputs.

    Parameters:
    ----------
    N : int, default=5000
        Total population (note: the model dynamics do not depend on N, just ensure N>0)
    I0_proportion : float, default=0.0002
        Initial proportion of infected individuals.
    beta : float, default=0.4
        Transmission rate.
    gamma : float, default=0.2
        Recovery rate, i.e. Reciprocal of the average time in infected state.
    beta_e : float, default=0
        Relative contagiousness of exposed individuals (compared to infected individuals).
    e : float, default=0.2
        Reciprocal of the average time in exposed state.
    c : float, default=2
        Behavioral response midpoint.
    k : float, default=16
        Behavioral response sensitivity.
    tau : float, default=0
        Delay time in reduction.
    dt : float, default=0.1
        Time step size.
    case : str
        Reduction mechanism:
          'Hill' : Hill function-based reduction. (default)
          'sigmoid' : Sigmoid-based reduction.
          'noReduction' : No reduction in transmission.
    t_end : float, default=500
        End time for the simulation.

    Returns:
    -------
    ts : array-like
        Time points of the simulation.
    results : array-like
        Normalized percentages of [S, E, I, R].
    reduction : array-like
        Reduction in transmission rate over time.
    Reffs : array-like
        Effective reproduction numbers over time.

    Notes:
    ------
    The function applies either a Hill or sigmoid reduction based on delayed prevalence. 
    For 'noReduction', no modification to transmission occurs.
    Normalized outputs are returned as percentages.
    """
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

    
    