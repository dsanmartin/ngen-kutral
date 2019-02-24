import numpy as np
from wildfire.fire import Fire
from wildfire import plots as p
#%%
# Asensio 2002 experiment
M, N = 128, 128
L = 5000 # Timesteps
dt = 1e-2 # dt
xa, xb = 0, 90 # x domain limit
ya, yb = 0, 90 # y domain limit
x = np.linspace(xa, xb, N) # x domain
y = np.linspace(ya, yb, M) # y domain
t = np.linspace(0, dt*L, L) # t domain

# Temperature initial condition
#u0 = lambda x, y: 6e0 * (np.exp(-5e-2*((x-20)**2 + (y-20)**2)) 
#  + np.exp(-5e-2*((x-15)**2 + (y-20)**2)) + np.exp(-5e-2*((x-20)**2 + (y-15)**2)))
#u0 = lambda x, y: 6e0 * np.exp(-5e-2*((x-20)**2 + (y-20)**2)) 
u0 = lambda x, y: 6e0 * np.exp(-5e-2*((x-20)**2 + (y-70)**2)) # JCC

# Fuel initial condition
np.random.seed(666)
b0 = lambda x, y: np.round(np.random.uniform(size=(x.shape)), decimals=2)#x*0 + 1 #S(x+.25, y+.25) #x*0 + 1

# Wind effect
gamma = 1
#v1 = lambda x, y, t: gamma * np.cos(np.pi/4 + x*0 + 1e-2*t) # 300
#v2 = lambda x, y, t: gamma * np.sin(np.pi/4 + x*0 + 1e-2*t) # 300
v1 = lambda x, y, t: gamma * np.cos(-np.pi/4 + x*0 + 1e-2*t) # 300 # JCC
v2 = lambda x, y, t: gamma * np.sin(-np.pi/4 + x*0 + 1e-2*t) # 300
V = (v1, v2)

# Parameters
kappa = 1e-1 # diffusion coefficient
epsilon = 3e-1#3e-2 # inverse of activation energy
upc = 3e0 # u phase change
q = 1 # reaction heat
alpha = 1e-3 # natural convection

#%% Mell 2006 experiment
M, N = 128, 128
L = 5000 # Timesteps
dt = 1e-2 # dt
xa, xb = -100, 100 # x domain limit
ya, yb = -100, 100 # y domain limit
x = np.linspace(xa, xb, N) # x domain
y = np.linspace(ya, yb, M) # y domain
t = np.linspace(0, dt*L, L) # t domain

# Temperature initial condition
#u0 = lambda x, y: 6e0*np.exp(-2e-2*((x+90)**2 + (y)**2)) #1e2
#u0 = lambda x, y: 6e1*(np.zeros((x.shape)) + np.ones((x.shape[0], 2)))
def u0(x, y):
  out = np.zeros((x.shape))
  #out[50:-50, :5] = 6*np.ones((x.shape[0]-100, 5))
  #out[20:-20, :5] = 6*np.ones((x.shape[0]-40, 5))
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
#%% TESTING TOPO
M, N = 128, 128
L = 1000 # Timesteps
dt = 1e-4 # dt
xa, xb = -1, 1 # x domain limit
ya, yb = -1, 1 # y domain limit
x = np.linspace(xa, xb, N) # x domain
y = np.linspace(ya, yb, M) # y domain
t = np.linspace(0, dt*L, L) # t domain

G = lambda x, y, s: np.exp(-1/s * (x**2 + y**2))
Gx = lambda x, y, s: -2/s * x * np.exp(-1/s * (x**2 + y**2))
Gy = lambda x, y, s: -2/s * y * np.exp(-1/s * (x**2 + y**2))

# Temperature initial condition
u0 = lambda x, y: 6e0 * G(x+.5, y-.5, 1e-2) #9

# Fuel initial condition
b0 = lambda x, y: 0.5 * G(x+.75, y-.75, .6) + 0.9 * G(x-.75, y+.75, 1) \
  + 0.016 * G(x+.65, y+.65, .3) + 0.015 * G(x-.65, y-.65, .7) #+ 0.8

#np.random.seed(666)
#b0 = lambda x, y: np.round(np.random.uniform(size=(x.shape)), decimals=2)

# Wind effect
gamma = 10
w1 = lambda x, y, t: gamma * np.cos(7/4 * np.pi + x*0)
w2 = lambda x, y, t: gamma * np.sin(7/4 * np.pi + y*0)
W = (w1, w2)

# Topography test
xi = 20
top = lambda x, y: 0.2 * G(x+.5, y+.5, .6) + 0.2 * G(x-.9, y-.9, .9)
t1 = lambda x, y: xi * (0.5 * G(x+.5, y +.6, .6) * Gx(x+.5, y+.5, .6) + 0.5 * G(x-.9, y-.9, .9) * Gx(x-.9, y-.9, .9))
t2 = lambda x, y: xi * (0.5 * G(x+.5, y+.5, .6) * Gy(x+.5, y+.5, .6) + 0.5 * G(x-.9, y-.9, .9) * Gy(x-.9, y-.9, .9))
T = (t1, t2)

# Vector
v1 = lambda x, y, t: w1(x, y, t) + t1(x, y)
v2 = lambda x, y, t: w2(x, y, t) + t2(x, y)
V = (v1, v2)

# Parameters
kappa = 1e-2 # diffusion coefficient
epsilon = 1e-1#3e-2 # inverse of activation energy
upc = 1e0 # u phase change
q = 5e-3 # reaction heat
alpha = 1e-3#1e-3 # natural convection
#%%
# Meshes for initial condition plots
X, Y = np.meshgrid(x, y)

# Plot initial conditions
#p.plotIC(X, Y, u0, b0, V, T=None, top=None)#=None, top=None)

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
    'complete': False,
    'components': (1, 1, 1)
}

ct = Fire(parameters)

#%%
# Finite difference in space
U, B = ct.solvePDE('fd', 'rk4')
#%%
p.plotJCC(t, X, Y, U, B, V)
#%%
tt, tim = 3000, '30'
UU = np.zeros((N*M, 3))
BB = np.zeros_like(UU)
UU[:,0] = X.flatten()
UU[:,1] = Y.flatten()
UU[:,2] = U[tt].flatten() 
BB[:,0] = X.flatten()
BB[:,1] = Y.flatten()
BB[:,2] = B[tt].flatten() 
np.savetxt("U" + tim + ".csv", UU, delimiter=" ", fmt='%.8f')
np.savetxt("B" + tim + ".csv", BB, delimiter=" ", fmt='%.8f')

#%%
p.plotComplete(t, X, Y, U, B, V, True)