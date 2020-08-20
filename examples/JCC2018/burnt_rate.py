#%%
import numpy as np
from wildfire import Fire
from wildfire.utils import old_plots as p
import matplotlib.pyplot as plt

#%% Mell 2006 experiment
Nx, Ny = 128, 128
Nt = 500 # Timesteps
xa, xb = -100, 100 # x domain limit
ya, yb = -100, 100 # y domain limit
ta, tb = 0, 30 # t domain
x = np.linspace(xa, xb, Nx) # x domain
y = np.linspace(ya, yb, Ny) # y domain
t = np.linspace(ta, tb, Nt + 1) # t domain

# Temperature initial condition
#u0 = lambda x, y: 6e0*np.exp(-2e-2*((x+90)**2 + (y)**2)) #1e2
#u0 = lambda x, y: 6e1*(np.zeros((x.shape)) + np.ones((x.shape[0], 2)))
def u0(x, y):
  out = np.zeros((x.shape))
#  #out[20:-20, :5] = 6*np.ones((x.shape[0]-40, 5))
  out[35:-35, :4] = 6*np.ones((x.shape[0]-70, 4))
#  #out[5:15, :4] = 6*np.ones((10, 4))
#  #out[-15:-5, :4] = 6*np.ones((10, 4))
  return out

# Fuel initial condition
#b0 = lambda x, y: x*0 + 1
b0 = lambda x, y: np.round(np.random.uniform(size=(x.shape)), decimals=2)

# Wind effect
gamma = 1
v1 = lambda x, y, t: gamma * np.cos(0 + x*0)
v2 = lambda x, y, t: gamma * np.sin(0 + x*0)
V = (v1, v2)

# Parameters
kappa = 1e1 # diffusion coefficient
epsilon = 3e-1#3e-2 # inverse of activation energy
upc = 1e0 # u phase change
q = 3#1 # reaction heat
alpha = 1e-2#1e-4 # natural convection
#%%
# Meshes for initial condition plots
X, Y = np.meshgrid(x, y)

# Plot initial conditions
p.plotIC(X, Y, u0, b0, V, T=None, top=None)

# Parameters for the model
parameters = {
    'kap': kappa, 
    'eps': epsilon, 
    'upc': upc, 
    'alp': alpha, 
    'q': q, 
    'x_lim': (xa, xb), 
    'y_lim': (ya, yb),
    't_lim': (ta, tb),
}

ct = Fire(**parameters)
#%%
# Finite difference in space
t, X, Y, U, B = ct.solvePDE(Nx, Ny, Nt, u0, b0, V, last=False)
#%%
p.plotJCC(t, X, Y, U, B, V, T=None, save=False)
#%% 
# BURNT RATE PLOT
plt.figure(figsize=(6, 4))

dt = t[1] - t[0]
dif_b = (B[1:] - B[:-1]) / dt
levels = np.arange(-0.04, np.max(dif_b), 0.01)

row = 5
tim = Nt // (row - 1)

for i in range(row):
    tt = i*tim
    if i == (row - 1):
        tt = -1
    plt.contourf(X, Y, dif_b[tt], levels=levels)

plt.xlabel(r"$x$", fontsize=16)
plt.ylabel(r"$y$", fontsize=16)
cbbr = plt.colorbar()
cbbr.set_label("Fuel consumption rate", size=14)
plt.show()
#plt.savefig('./experiments/JCC2018/burnt_rate.eps', format='eps', dpi=200, transparent=True, bbox_inches='tight', pad_inches=0)
