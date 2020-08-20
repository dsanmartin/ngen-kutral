import numpy as np
from wildfire import Fire
from wildfire.utils.functions import G
from wildfire.utils import old_plots as p
from wildfire.utils import plots
import matplotlib.pyplot as plt

# Temperature initial condition
u0 = lambda x, y: 6 * G(x-20, y-20, 20)

# Fuel initial condition
np.random.seed(666)
br = lambda x, y: np.round(np.random.uniform(size=(x.shape)), decimals=2)
def b0(x, y):
    B = br(x, y)
    B[0,:] = np.zeros(len(x))
    B[-1,:] = np.zeros(len(x))
    B[:,0] = np.zeros(len(y))
    B[:,-1] = np.zeros(len(y))
    return B

# Wind effect
gamma = 1
w1 = lambda x, y, t: gamma * np.cos(np.pi/4 + x * 0)
w2 = lambda x, y, t: gamma * np.sin(np.pi/4 + x * 0)

V = (w1, w2)

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
X, Y = np.meshgrid(np.linspace(0, 90, Nx), np.linspace(0, 90, Ny))
p.plotIC(X, Y, u0, b0, (w1, w2), None, None, save=False)

t, X, Y, Ufft, Bfft = Fire(**parameters).solvePDE(Nx, Ny, Nt, u0, b0, V, space_method='fft')

plots.UB(t, X, Y, Ufft, Bfft, V)
