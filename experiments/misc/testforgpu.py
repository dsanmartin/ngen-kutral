import numpy as np
from wildfire.fire import Fire
from wildfire import plots as p
#%%
# Asensio 2002 experiment
M, N = 128, 128
L = 5000 # Timesteps
dt = 1e-3 # dt
xa, xb = 0, 90 # x domain limit
ya, yb = 0, 90 # y domain limit
x = np.linspace(xa, xb, N) # x domain
y = np.linspace(ya, yb, M) # y domain
t = np.linspace(0, dt*L, L) # t domain

# Temperature initial condition
u0 = lambda x, y: 6e0*np.exp(-5e-2*((x-45)**2 + (y-45)**2)) 

# Fuel initial condition
np.random.seed(666)
b0 = lambda x, y: np.round(np.random.uniform(size=(x.shape)), decimals=2)
#b0 = lambda x, y: x*0

# Wind effect
gamma = 1
v1 = lambda x, y, t: gamma * np.cos(np.pi/4 + x*0) # 300
v2 = lambda x, y, t: gamma * np.sin(np.pi/4 + x*0) # 300
V = (v1, v2)

# Parameters
kappa = 1e-0 # diffusion coefficient
epsilon = 3e-1#3e-2 # inverse of activation energy
upc = 3e0 # u phase change
q = 1 # reaction heat
alpha = 1e-3 # natural convection

#%%
# Meshes for initial condition plots
X, Y = np.meshgrid(x, y)

# Plot initial conditions
p.plotIC(X, Y, u0, b0, V, T=None, top=None)

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
    'complete': False
}

ct = Fire(parameters)

#%%
# Finite difference in space
U, B = ct.solvePDE('fd', 'euler')
#%%
# Plot approximations
for i in range(L):
  if i%500 != 0: continue
  p.plotUBs(i, t, X, Y, U, B, V)
