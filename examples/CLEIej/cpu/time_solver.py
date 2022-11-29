import numpy as np
from scipy.integrate import solve_ivp

# Euler method
def Euler(t, y0, f):
    Nt = t.shape[0]
    y = np.zeros((Nt, y0.shape[0]))
    y[0] = y0
    dt = t[1] - t[0]
    for n in range(Nt - 1):
        y[n+1] = y[n] + dt * f(t[n], y[n])     
    return y

# Fourth-order Runge Kutta method
def RK4(t, y0, f):
    Nt = t.shape[0]
    y = np.zeros((Nt, y0.shape[0]))
    y[0] = y0
    dt = t[1] - t[0]
    for n in range(Nt - 1):
        k1 = f(t[n], y[n])
        k2 = f(t[n] + 0.5 * dt, y[n] + 0.5 * dt * k1)
        k3 = f(t[n] + 0.5 * dt, y[n] + 0.5 * dt * k2)
        k4 = f(t[n] + dt, y[n] + dt * k3)
        y[n + 1] = y[n] + (1/6) * dt * (k1 + 2 * k2 + 2 * k3 + k4)
    return y
    
# Adams-Bashforth s=5
def AB5(t, y0, f):
    Nt = t.shape[0]
    y = np.zeros((Nt, y0.shape[0]))
    y[0] = y0
    dt = t[1] - t[0]
    for n in range(Nt - 5):
        fn = f(t[n], y[n])
        y[n+1] = y[n] + dt * fn
        fn1 = f(t[n+1], y[n+1])
        y[n+2] = y[n+1] + dt / 2 * (3 * fn1 - fn)
        fn2 = f(t[n+2], y[n+2])
        y[n+3] = y[n+2] + dt / 12 * (23 * fn2 - 16 * fn1 + 5 * fn)
        fn3 = f(t[n+3], y[n+3])
        y[n+4] = y[n+3] + dt / 24 * (55 * fn3 - 59 * fn2 + 37 * fn1 - 9 * fn)
        fn4 = f(t[n+4], y[n+4])
        y[n+5] = y[n+4] + dt / 720 * (1901 * fn4 - 2774 * fn3 + 2616 * fn2 - 1274 * fn1 + 251 * fn)
    return y

# IVP using solve_ivp from scipy https://docs.scipy.org/doc/scipy/reference/generated/scipy.integrate.solve_ivp.html
def IVP(t, y0, f, method):
    sol = solve_ivp(f, (t[0], t[-1]), y0, t_eval=t, method=method)
    return sol.y.T