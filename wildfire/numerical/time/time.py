"""Time approximation using numerical integration methods.

Solve Initial Value Problem (IVP) or ODE:
    .. math::
        \frac{dy}{dt} = F(t, y) \\
        y(t_0) = y_0

Implements the following integration methods:
    - Euler method
    - Runge-Kutta of fourth order method (RK4)

Also includes `solve_ivp` from `scipy.integrate`. 

Details in: 
    - Sauer, T. (2018). Numerical Analysis. Pearson. https://www.pearson.com/us/higher-education/program/Sauer-Numerical-Analysis-3rd-Edition/PGM1735484.html
    - Scipy. solve_ivp. https://docs.scipy.org/doc/scipy/reference/generated/scipy.integrate.solve_ivp.html
"""
import numpy as np
from scipy.integrate import solve_ivp 

def Euler(t, F, y0, last=True):
    """Euler method implementation.

    Parameters
    ----------
    t : array_like
        Time discrete variable.
    F : function
        RHS function of ODE.
    y0 : array_like
        Initial condition
    last : bool, optional
        Return and keep only last approximation, by default True.

    Returns
    -------
    array_like
        Aproximated solution of ODE.

    """
    # Get number time of nodes
    Nt = t.shape[0]

    # Get :math:`\Delta t`
    dt = t[1] - t[0] 

    # Only keep and return last approximation
    if last: 
        y = y0
    
        for k in range(Nt - 1):
            yc = np.copy(y)
            y = yc + dt * F(t[k], yc)

    # Keep and return array with all approximations
    else:
        y = np.zeros((Nt, y0.shape[0])) 
        y[0] = y0 # Initial condition
        
        for k in range(Nt - 1):
            y[k+1] = y[k] + dt * F(t[k], y[k])
        
    return y

def RK4(t, F, y0, last=True):
    """Runge-Kutta of fourth order implementation.

    Parameters
    ----------
    t : array_like
        Time discrete variable.
    F : function
        RHS function of ODE.
    y0 : array_like
        Initial condition
    last : bool, optional
        Return and keep only last approximation, by default True.

    Returns
    -------
    array_like
        Aproximated solution of ODE.

    """
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

def IVP(t, F, y0, last=True, method='RK45'):
    """Solve IVP wrapper.

    Parameters
    ----------
    t : array_like
        Time discrete variable.
    F : function
        RHS function of ODE.
    y0 : array_like
        Initial condition
    last : bool, optional
        Return and keep only last approximation, by default True.
    method : strin, optional
        Numerical method to solve IVP, default RK45. 
        Also includes 'RK23', 'DOP853', 'Radau', 'BDF', 'LSODA'. More details in `scipy.integrate.solve_ivp` documentation.

    Returns
    -------
    array_like
        Aproximated solution of ODE.

    """
    t_min = t[0]
    t_max = t[-1]
    if last:
        t_eval = np.array([t_max])
    else:
        t_eval = t
    sol = solve_ivp(F, (t_min, t_max), y0, t_eval=t_eval, method=method)    
    return sol.y