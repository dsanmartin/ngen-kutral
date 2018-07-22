# Convergence Analysis with Asension 2002 Experiment
import numpy as np
from wildfire.fire import Fire

# For reproducibility of random fuel
np.random.seed(666)

M = 4096
N = M
dt_dir = '1e-10'
L = 500 # Timesteps
dt = float(dt_dir) # dt
xa, xb = 0, 90 # x domain limit
ya, yb = 0, 90 # y domain limit
x = np.linspace(xa, xb, N) # x domain
y = np.linspace(ya, yb, M) # y domain
t = np.linspace(0, dt*L, L) # t domain

# Temperature initial condition
u0 = lambda x, y: 6e0*np.exp(-5e-2*((x-20)**2 + (y-20)**2)) 

# Fuel initial condition
b0 = lambda x, y: np.round(np.random.uniform(size=(x.shape)), decimals=2)

# Wind effect
gamma = 1
v1 = lambda x, y, t: gamma * np.cos(np.pi/4 + x*0) 
v2 = lambda x, y, t: gamma * np.sin(np.pi/4 + x*0)
V = (v1, v2)

# Parameters
kappa = 1e-1 # diffusion coefficient
epsilon = 3e-1 # inverse of activation energy
upc = 3e0 # u phase change
q = 1 # reaction heat
alpha = 1e-3 # natural convection

#M, N = 2048, 2048
#x = np.linspace(xa, xb, N) # x domain
#y = np.linspace(ya, yb, M) # y domain

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

# Solve PDE, keep only last approximation
U, B = ct.solvePDE('fd', 'last')

# Save last temperature and fuel
exp_dir = str(L) + "/" + dt_dir + "/"
np.save('convergence/' + exp_dir + 'U_' + str(M), U)
np.save('convergence/' + exp_dir + 'B_' + str(M), B)
