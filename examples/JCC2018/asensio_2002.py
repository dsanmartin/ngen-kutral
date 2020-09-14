import numpy as np
from wildfire import Fire
from wildfire.utils import old_plots as p
#%%
# Asensio 2002 experiment
Nx, Ny = 128, 128
Nt = 500 # Timesteps
xa, xb = 0, 90 # x domain limit
ya, yb = 0, 90 # y domain limit
ta, tb = 0, 30
x = np.linspace(xa, xb, Nx) # x domain
y = np.linspace(ya, yb, Ny) # y domain
t = np.linspace(ta, tb, Nt + 1) # t domain

# Temperature initial condition
u0 = lambda x, y: 6e0*np.exp(-5e-2*((x-20)**2 + (y-20)**2)) 

# Fuel initial condition
np.random.seed(666)
b0 = lambda x, y: np.round(np.random.uniform(size=(x.shape)), decimals=2)#x*0 + 1 #S(x+.25, y+.25) #x*0 + 1

# Wind effect
gamma = 1
v1 = lambda x, y, t: gamma * np.cos(np.pi/4 + x*0) # 300
v2 = lambda x, y, t: gamma * np.sin(np.pi/4 + x*0) # 300
V = lambda x, y, t: (v1(x, y, t), v2(x, y, t)) #V = (v1, v2)

# Parameters
kappa = 1e-1 # diffusion coefficient
epsilon = 3e-1#3e-2 # inverse of activation energy
upc = 3 # u phase change
q = 1 # reaction heat
alpha = 1e-3 # natural convection

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
#p.plotJCCCols(t, X, Y, U, B, V, T=None, save=True)
#%%
# UU = np.zeros((N*M, 3))
# BB = np.zeros_like(UU)
# UU[:,0] = X.flatten()
# UU[:,1] = Y.flatten()
# UU[:,2] = U[-1].flatten() 
# BB[:,0] = X.flatten()
# BB[:,1] = Y.flatten()
# BB[:,2] = B[-1].flatten() 
#np.savetxt("U30.csv", UU, delimiter=" ", fmt='%.8f')
#np.savetxt("B30.csv", BB, delimiter=" ", fmt='%.8f')