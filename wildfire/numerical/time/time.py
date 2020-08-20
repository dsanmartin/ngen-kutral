"""Time approximation using numerical integration methods.

Implements integration methods:
    - Euler method
    - Runge-Kutta of fourth order method (RK4)

Details in: Sauer, T. (2018). Numerical Analysis. Pearson	.
https://www.pearson.com/us/higher-education/program/Sauer-Numerical-Analysis-3rd-Edition/PGM1735484.html
"""
import numpy as np
from scipy.integrate import solve_ivp 

# Euler method #
def Euler(t, F, y0, last=True):

    # Get number time of nodes
    Nt = t.shape[0]

    # Get \Delta t
    dt = t[1] - t[0] 

    if last: # Only keep and return last approximation
        y = y0
    
        for k in range(Nt - 1):
            yc = np.copy(y)
            y = yc + dt * F(t[k], yc)

    else:
        # Array with approximations
        y = np.zeros((Nt, y0.shape[0])) 
        y[0] = y0 # Initial condition
        
        for k in range(Nt - 1):
            y[k+1] = y[k] + dt * F(t[k], y[k])
        
    return y

# Keep last approximation
def EulerLast(t, F, y0):    
    # Get number time of nodes
    Nt = t.shape[0]

    # Get \Delta t
    dt = t[1] - t[0] 

    y = y0
    
    for k in range(Nt - 1):
        yc = np.copy(y)
        y = yc + dt * F(t[k], yc)
        
    return y

# Runge-Kutta 4th order #
# Keep all approximations
def RK4(t, F, y0, last=True):

    # Get number time of nodes
    Nt = t.shape[0]

    # Get \Delta t
    dt = t[1] - t[0] 
    
    if last: # Only keep and return last approximation
        # Initial condition
        y = y0
        
        for k in range(Nt - 1):
            yc = np.copy(y)
            k1 = F(t[k], yc)
            k2 = F(t[k] + 0.5 * dt, yc + 0.5 * dt * k1)
            k3 = F(t[k] + 0.5 * dt, yc + 0.5 * dt * k2)
            k4 = F(t[k] + dt, yc + dt * k3)

            y = yc + (1/6) * dt * (k1 + 2 * k2 + 2 * k3 + k4)

    else: # Keep and return all approximations
        # Array for approximations
        y = np.zeros((Nt, y0.shape[0]))
        y[0] = y0 # Initial condition
        
        for k in range(Nt - 1):
            k1 = F(t[k], y[k])
            k2 = F(t[k] + 0.5 * dt, y[k] + 0.5 * dt * k1)
            k3 = F(t[k] + 0.5 * dt, y[k] + 0.5 * dt * k2)
            k4 = F(t[k] + dt, y[k] + dt * k3)

            y[k + 1] = y[k] + (1/6) * dt * (k1 + 2 * k2 + 2 * k3 + k4)
        
    return y

# Keep just last approximation
def RK4Last(t, F, y0):

    # Get number time of nodes
    Nt = t.shape[0]

    # Get \Delta t
    dt = t[1] - t[0] 

    # Initial condition
    y = y0
    
    for k in range(Nt - 1):
        yc = np.copy(y)
        k1 = F(t[k], yc)
        k2 = F(t[k] + 0.5 * dt, yc + 0.5 * dt * k1)
        k3 = F(t[k] + 0.5 * dt, yc + 0.5 * dt * k2)
        k4 = F(t[k] + dt, yc + dt * k3)

        y = yc + (1/6) * dt * (k1 + 2 * k2 + 2 * k3 + k4)
        
    return y

def IVP(t, F, y0):
    t_min = t[0]
    t_max = t[-1]
    sol = solve_ivp(F, (t_min, t_max), y0, t_eval=[t_max], method='RK45')    
    return sol.y