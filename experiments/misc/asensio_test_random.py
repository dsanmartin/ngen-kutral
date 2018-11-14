import numpy as np
from wildfire.fire import Fire
from wildfire import plots as p

"""
Trying to copy Asensio 2002 experiment
using complete PDE system, K(u) != k
"""
M, N = 128, 128
L = 500 # Timesteps
dt = 1e-2 # dt
xa, xb = 0, 300 # x domain limit
ya, yb = 0, 300 # y domain limit
x = np.linspace(xa, xb, N) # x domain
y = np.linspace(ya, yb, M) # y domain
t = np.linspace(0, dt*L, L) # t domain

# Dimensional parameters
T_inf = 300 # kelvin
t_0 = 8987 # seconds
l_0 = 0.3 # meters

# Temperature initial condition
u0 = lambda x, y: 4.8e0*np.exp(-5e-3*((x-75)**2 + (y-75)**2)) 

# Fuel initial condition
b0 = lambda x, y: np.round(np.random.uniform(size=(x.shape)), decimals=2)#x*0 + 1 #S(x+.25, y+.25) #x*0 + 1

# Wind effect
gamma = 1e-2#e-3
w1 = lambda x, y, t: gamma * 300 + x*0 
w2 = lambda x, y, t: gamma * 300 + y*0 
W = (w1, w2)

# Vector
v1 = lambda x, y, t: w1(x, y, t)
v2 = lambda x, y, t: w2(x, y, t)
V = (v1, v2)

# Parameters
kappa = 1e-1 # diffusion coefficient
epsilon = 3e-1 # inverse of activation energy
upc = 1 # 1 # u phase change # Smaller upc -> bigger fire front
q = 1 # reaction heat
alpha = 1e-3 # natural convection

# Meshes for initial condition plots
#X, Y = np.meshgrid(x, y)

# Plot initial conditions
#p.plotIC(X, Y, u0, b0, V, W, T=None, top=None)

# Parameters for the model
parameters = {
    'u0': u0, 
    'beta0': b0,
    'v': V,
    'kappa': kappa, 
    'epsilon': epsilon, 
    'upc': upc, 
    'q': q, 
    'alpha': alpha, 
    'x': x, 
    'y': y,
    't': t,
    'sparse': True,
    'show': False,
    'complete': True
}

ct = Fire(parameters)


# Finite difference in space
U, B = ct.solvePDE('fd', 'random', .005, .005)

#ct.plots(U, B)

# PLOT JCC
X, Y = np.meshgrid(x, y)
p.plotJCC(t, X, Y, U, B, W, T=None, save=False)
