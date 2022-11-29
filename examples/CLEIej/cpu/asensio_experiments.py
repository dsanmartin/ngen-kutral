# Library imports
import numpy as np
import matplotlib.pyplot as plt
import utils
from time_solver import *
import time

# Function for boundary conditions
def boundaryConditions(U, B):
    Ub = np.copy(U)
    Bb = np.copy(B)
    Ny, Nx = Ub.shape
    # Only Dirichlet: 
    # Temperature
    Ub[ 0,:] = np.zeros(Nx)
    Ub[-1,:] = np.zeros(Nx)
    Ub[:, 0] = np.zeros(Ny)
    Ub[:,-1] = np.zeros(Ny)
    # Fuel
    Bb[0 ,:] = np.zeros(Nx)
    Bb[-1,:] = np.zeros(Nx)
    Bb[:, 0] = np.zeros(Ny)
    Bb[:,-1] = np.zeros(Ny)

    return Ub, Bb

# Rigth hand side of the equations
def RHS(t, r, **kwargs):
    # Parameters 
    X, Y = kwargs['x'], kwargs['y']
    x, y = X[0], Y[:, 0] 
    V   = kwargs['V']
    kap = kwargs['kap']
    f = kwargs['f']
    g = kwargs['g']
    Nx = x.shape[0]
    Ny = y.shape[0]
    dx = x[1] - x[0]
    dy = y[1] - y[0]

    # Vector field evaluation
    V1, V2 = V(x, y, t)
    
    # Recover u and b from vector. Reshape them into matrices
    U = np.copy(r[:Ny * Nx].reshape((Ny, Nx)))
    B = np.copy(r[Ny * Nx:].reshape((Ny, Nx)))

    # Compute derivatives #
    Ux = np.zeros_like(U)
    Uy = np.zeros_like(U)
    Uxx = np.zeros_like(U)
    Uyy = np.zeros_like(U)
    # # First derivatice (forward finite difference)
    # Ux[1:-1, 1:-1] = (U[1:-1, 1:-1] - U[1:-1, :-2]) / dx
    # Uy[1:-1, 1:-1] = (U[1:-1, 1:-1] - U[:-2, 1:-1]) / dy
    # First derivatives (central finite difference)
    Ux[1:-1, 1:-1] = (U[1:-1, 2:] - U[1:-1, :-2]) / 2 / dx
    Uy[1:-1, 1:-1] = (U[2:, 1:-1] - U[:-2, 1:-1]) / 2 / dy
    # Second derivatives (central finite difference)
    Uxx[1:-1, 1:-1] = (U[1:-1, 2:] - 2 * U[1:-1, 1:-1] + U[1:-1, :-2]) / dx / dx
    Uyy[1:-1, 1:-1] = (U[2:, 1:-1] - 2 * U[1:-1, 1:-1] + U[:-2, 1:-1]) / dy / dy
        
    # Laplacian of u
    lapU = Uxx + Uyy

    # Compute diffusion term
    diffusion = kap * lapU # \kappa \Delta u
    # Compute convection term
    convection = Ux * V1 + Uy * V2 # v \cdot grad u.    
    # Compute reaction term
    reaction = f(U, B) # eval fuel
    
    # Compute RHS
    Uf = diffusion - convection + reaction # Temperature
    Bf = g(U, B) # Fuel
    
    # Add boundary conditions
    Uf, Bf = boundaryConditions(Uf, Bf)

    # Build \mathbf{y} = [vec(u), vec(\beta)]^T and return
    return np.r_[Uf.flatten(), Bf.flatten()] 

### PARAMETERS ###
# Model parameters #
kap = 1e-1
eps = 3e-1
upc = 3
alp = 1e-3
q = 1
x_min, x_max = 0, 90
y_min, y_max = 0, 90
t_min, t_max = 0, 10
## Number of experiments
N_exp = 10

# Re-define PDE funtions with parameters #
s = lambda u: utils.H(u, upc)
ff = lambda u, b: utils.f(u, b, eps, alp, s)
gg = lambda u, b: utils.g(u, b, eps, q, s)

# Initial conditions
u0 = lambda x, y: 6 * utils.G(x - 20, y - 20, 20) # Temperature
b0 = lambda x, y: x * 0 + 1 # Fuel
# Wind effect
w1 = lambda x, y, t: np.cos(np.pi/4 + x * 0) # Wind 
w2 = lambda x, y, t: np.sin(np.pi/4 + x * 0) # Wind
V = lambda x, y, t: (w1(x, y, t), w2(x, y, t)) # Vector field

# Numerical #
# Time nodes
Nt = 500

# Space nodes
Ns = np.array([64, 128, 256, 512])
for N in Ns:
    Nx = Ny = N

    # Domain #
    x = np.linspace(x_min, x_max, Nx)
    y = np.linspace(y_min, y_max, Ny)
    t = np.linspace(t_min, t_max, Nt)
    X, Y = np.meshgrid(x, y)
    dx, dy, dt = x[1] - x[0], y[1] - y[0], t[1] - t[0]

    # Just log
    print("Nx: %d, Ny: %d, Nt: %d" % (Nx, Ny, Nt))
    print("dx: %.2f, dy: %.2f, dt: %.2f" % (dx, dy, dt))

    # Parameters #
    # Domain, functions, etc.
    params = {'x': X, 'y': Y, 'V': V, 'kap': kap, 'f': ff, 'g': gg,}

    # Initial condition (vectorized)
    y0 = np.r_[u0(X, Y).flatten(), b0(X, Y).flatten()]

    # Mask RHS to include parameters
    F = lambda t, y: RHS(t, y, **params)

    # Solve IVP #
    time_start = time.time()
    # R = IVP(t, y0, F, 'RK45') # Use solve_ivp from scipy.integrate 
    for e in range(N_exp):
        R = RK4(t, y0, F) # Use RK4 'from scratch'
    time_end = time.time()
    solve_time = (time_end - time_start) / N_exp
    print("Time: ", solve_time, "[s]\n")

