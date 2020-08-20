import numpy as np
from wildfire import Fire
from wildfire.utils.functions import G
from wildfire.utils import plots
from wildfire.utils import old_plots as p
import matplotlib.pyplot as plt

# Temperature initial condition
u0 = lambda x, y: 7 * G(x-20, y-20, 20) + 4 * G(x-80, y-70, 20) + 4 * G(x-20, y-35, 50)

# Fuel initial condition
b0 = lambda x, y: G(x-10, y-10, 200) + G(x-20, y-60, 300) + G(x-45, y-45, 200) + G(x-75, y-45, 300) + G(x-80, y-80, 100)

# Terrain effect 
T = lambda x, y: 1.5 * (3 * G(x-45, y-45, 40) + 2 * G(x-30, y-30, 60) + 3 * G(x-70, y-70, 60) + 2 * G(x-20, y-70, 70))
Tx = lambda x, y: -2 * 1.5 * ( 3 * (x-45) * G(x-45, y-45, 40) / 40 + 2 * (x-30) * G(x-30, y-30, 60) / 60 + 3 * (x-70) * G(x-70, y-70, 60) / 60 + 2 * (x-20) * G(x-20, y-70, 70) / 70)
Ty = lambda x, y: -2 * 1.5 * ( 3 * (y-45) * G(x-45, y-45, 40) / 40 + 2 * (y-30) * G(x-30, y-30, 60) / 60 + 3 * (y-70) * G(x-70, y-70, 60) / 60 + 2 * (y-70) * G(x-20, y-70, 70) / 70) 

X, Y = np.meshgrid(np.linspace(0, 90, 2 ** 7), np.linspace(0, 90, 2 ** 7))

# Wind effect
gamma = 1
w1 = lambda x, y, t: gamma * np.cos(np.pi/4 + x * 0 + t * 0.0025) 
w2 = lambda x, y, t: gamma * np.sin(np.pi/4 + x * 0 + t * 0.025) 

rm_t = 1 # Set to 0 for no terrain

# Vector field w(x,y,t) + \nabla T(x,y)
v1 = lambda x, y, t: w1(x, y, t) + rm_t * Tx(x, y)
v2 = lambda x, y, t: w2(x, y, t) + rm_t * Ty(x, y)
V = (v1, v2)

e = 7
Nx = 2 ** e
Ny = 2 ** e
Nt = 100

# Parameters for the model
parameters = {    
    # Physical
    'kap': 1e-1, # diffusion coefficient
    'eps': 3e-1, # inverse of activation energy
    'upc': 3, # u phase change
    'alp': 1e-3, # natural convection
    'q': 1, # reaction heat
    'x_lim': (0, 90), 
    'y_lim': (0, 90), 
    't_lim': (0, 20),
}

# Meshes for initial condition plots
p.plotIC(X, Y, u0, b0, (w1, w2), (Tx, Ty), T, save=False)

ct = Fire(**parameters)

t, X, Y, U, B = ct.solvePDE(Nx, Ny, Ny, u0, b0, V, space_method='fft', last=False)

for k in range(Nt):
    if k % 10 == 0:
        plots.UBs(k, t, X, Y, U, B, V)
