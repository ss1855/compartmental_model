
#!/usr/bin/env python
# coding: utf-8

import os
import numpy as np
from numpy import exp
import matplotlib.pyplot as plt
from lmfit import minimize, Parameters, Parameter, report_fit
import pandas as pd
import math

# Define the SEIRD model
def Model(x, params):
    beta = params['betasymptomatic']
    delta = params['delta']
    gamma = params['gamma']
    lambd = params['lambd']
    N = 1.4e9
    
    xdot = np.array([
        - (beta * x[0] * (x[2] / N)),
        (beta * x[0] * (x[2] / N)) - lambd * x[1],
        lambd * x[1] - (gamma + delta) * x[2],
        (gamma * x[2]),
        delta * x[2],
        lambd * x[1],
        gamma * x[2],
        delta * x[2]
    ])
    return xdot

# Runge-Kutta 4th order method for integration
def RungeKutta4(f, x0, t0, tf, dt):
    t = np.arange(t0, tf, dt)
    nt = t.size
    nx = x0.size
    x = np.zeros((nx, nt))
    x[:, 0] = x0
    
    for k in range(nt-1):
        k1 = dt * f(t[k], x[:, k])
        k2 = dt * f(t[k] + dt / 2, x[:, k] + k1 / 2)
        k3 = dt * f(t[k] + dt / 2, x[:, k] + k2 / 2)
        k4 = dt * f(t[k] + dt, x[:, k] + k3)
        dx = (k1 + 2 * k2 + 2 * k3 + k4) / 6
        x[:, k+1] = x[:, k] + dx
    return x, t

# Function to calculate error
def error(params, x0, t0, tf, dt, data):
    f = lambda t, x: Model(x, params)
    x, t = RungeKutta4(f, x0, t0, tf, dt)
    return (x.T[:, 5:8] - data).ravel()

# Set model parameters
def set_parameters():
    beta = 0.20
    delta = 0.00163
    gamma = 1/10
    lambd = 1/3.5
    params = Parameters()
    params.add("betasymptomatic", value=beta, vary=True, min=0, max=1)
    params.add("lambd", value=lambd, vary=True, min=1/2, max=1/5)
    params.add("delta", value=delta, vary=True, min=0, max=0.1)
    params.add('gamma', value=gamma, vary=True, min=1/5, max=1/30)
    return params

# Read data from an Excel file
def read_data(file_path):
    return pd.read_excel(file_path)

# Main function
def main():
    opt = read_data("C:/Users/Sujan Shrestha/Desktop/graphtoday.xlsx")
    df = pd.DataFrame(opt)
    
    x0 = np.array([1.4e9 - 30 - 88 - 10 - 2, 30, 88, 10, 2, 100, 10, 2])
    tf = [11, 37, 51, 79, 140, 191, 201, 215, 227, 253, 293]
    t0 = 0
    delta1 = []
    gamma1 = []
    lambd1 = []
    beta1 = []

    for a in tf:
        tf = a
        dt = 1
        data = df.loc[t0: (tf - 1), ['Average infected', 'Average Recovered', 'Average Death']].values
        result = minimize(error, set_parameters(), args=(x0, t0, tf, dt, data), method='least_squares')
        beta = result.params['betasymptomatic'].value
        delta = result.params['delta'].value
        gamma = result.params['gamma'].value
        lambd = result.params['lambd'].value
        params1 = set_parameters()
        params1.add('lambd', value=lambd, vary=False)
        params1.add('delta', value=delta, vary=False)
        params1.add('gamma', value=gamma, vary=False)
        params1.add('betasymptomatic', value=beta, vary=False)
        delta1.append(delta)
        gamma1.append(gamma)
        lambd1.append(lambd)
        beta1.append(beta)
        
        seir = lambda t, x: Model(x, params1)
        x, t = RungeKutta4(seir, x0, t0, tf, dt)
        num = tf - t0 - 1
        t0 = tf - 1
        x0 = np.array([x[0, num], x[1, num], x[2, num], x[3, num], x[4, num], x[5, num], x[6, num], x[7, num]])
    
    b = np.array(beta1).reshape(1, 11)
    c = np.array(gamma1).reshape(1, 11)
    d = np.array(delta1).reshape(1, 11)
    e = np.array(lambd1).reshape(1, 11)
    par = np.concatenate([b, c, d, e], axis=0).T
    return par
if __name__ == "__main__":
    par = main()
