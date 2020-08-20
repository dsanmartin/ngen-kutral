import numpy as np
import wildfire
from wildfire.utils.functions import G
from wildfire.utils import plots

### PARAMETERS ###
# Model parameters #
kap = 1e-1
eps = 3e-1
upc = 3
alp = 1e-3
q = 1
x_min, x_max = 0, 90
y_min, y_max = 0, 90
t_min, t_max = 0, 30

# Numerical #
# Space
space_method = 'fd'
Nx = 128
Ny = 128
acc = 2
sparse = False

# Time
time_method = 'RK4'
Nt = 100
last = True

# Initial conditions
u0 = lambda x, y: 6 * G(x-20, y-20, 20)

# Fuel initial condition
def b0(x, y):
  np.random.seed(666)
  br = lambda x, y: np.round(np.random.uniform(size=(x.shape)), decimals=2)
  B = br(x, y)
  x_rows, x_cols = x.shape
  y_rows, y_cols = y.shape
  B[0,:] = np.zeros(x_cols)
  B[-1,:] = np.zeros(x_cols)
  B[:,0] = np.zeros(y_rows)
  B[:,-1] = np.zeros(y_rows)
  return B

# Wind effect
gamma = 1
w1 = lambda x, y, t: gamma * np.cos(np.pi/4 + x * 0)
w2 = lambda x, y, t: gamma * np.sin(np.pi/4 + x * 0)
V = (w1, w2)
### PARAMETERS ###

# Physical parameters for the model
physical_parameters = {    
    'kap': kap, # diffusion coefficient
    'eps': eps, # inverse of activation energy
    'upc': upc, # u phase change
    'q': q, # reaction heat
    'alp': alp, # natural convection,
    # Domain
    'x_lim': (x_min, x_max), # x-axis domain 
    'y_lim': (y_min, y_max), # y-axis domain
    't_lim': (t_min, t_max) # time domain
}

wildfire_ = wildfire.Fire(**physical_parameters)
t, X, Y, U, B = wildfire_.solvePDE(Nx, Ny, Nt, u0, b0, V, space_method, time_method, acc=acc, sparse=sparse)

## Plot last results.
plots.UB(t, X, Y, U, B, V)
