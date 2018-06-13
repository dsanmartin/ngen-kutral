import wildfire
import numpy as np
import matplotlib.pyplot as plt

# Helper to build reaction rate
# Gaussian basis
def G(x, y):
  return np.exp(-((x)**2 + (y)**2))

# Superposition of gaussians based in https://commons.wikimedia.org/wiki/File:Scalar_field.png
def S(x, y):
  return G(2*x, 2*y) + 0.8 * G(2*x + 1.25, 2*y + 1.25) + 0.5 * G(2*x - 1.25, 4*y + 1.25) \
    - 0.5 * G(3*x - 1.25, 3*y - 1.25) + 0.35 * G(2*x + 1.25, 2*y - 1.25) \
    + 0.8 * G(x - 1.25, 3*y + 1.5) + 1.2 * G(x + 1.25, 3*y - 1.85)

def plotField(Xv, Yv, V):
  plt.quiver(Xv, Yv, V[0](Xv, Yv), V[1](Xv, Yv))  
  plt.title("Wind")
  plt.show()
  
def plotScalar(X, Y, U, title, cmap_):
  plt.imshow(U(X,Y), origin="lower", cmap=cmap_, 
             extent=[X[0,0], X[-1, -1], Y[0, 0], Y[-1, -1]])
  plt.title(title)
  plt.colorbar()
  plt.show()
  
# Domain: [-1, 1]^2 x [0, T]
M, N = 50, 50 # Resolution
T = 1500 # Max time
dt = 1e-3 # Timestep
xa, xb = -1, 1 # x domain limit
ya, yb = -1, 1 # y domain limit
x = np.linspace(xa, xb, N) # x domain
y = np.linspace(ya, yb, M) # y domain
t = np.linspace(0, dt*T, T) # t domain

# Meshes for initial condition plots
X, Y = np.meshgrid(x, y)
Xv, Yv = np.mgrid[xa:xb:complex(0, N // 4), ya:yb:complex(0, M // 4)]

# TODO: define parameters in function of real ones
#T_env = 300
#Ea = 83.68
#A = 1e9
#rho = 1e2
#C = 1
#k = 1

# Vector field V = (v1, v2). "Incompressible flow div(V) = 0"
gamma = 1
v1 = lambda x, y: gamma * np.cos(y) 
v2 = lambda x, y: gamma * np.sin(x)
V = (v1, v2)

# Lambda function for temperature initial condition
u0 = lambda x, y: 1e1*np.exp(-40*((x+.8)**2 + (y-.8)**2)) 

# Lambda function for fuel initial condition
b0 = lambda x, y: x*0 + 1 #S(x+.25, y+.25) #x*0 + 1

# Non dimensional parameters
kappa = 1*1e-3 # diffusion coefficient
epsilon = 1*1e-1 # inverse of activation energy
upc = 1*.1 # u phase change
q = 1*1e-1 # reaction heat
alpha = 1e-2 # natural convection

# Plot initial conditions
plotField(Xv, Yv, V)
plotScalar(X, Y, b0, "Fuel", plt.cm.Oranges)
plotScalar(X, Y, u0, "Initial contidion", plt.cm.jet)

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
    't': t
}
#%%
# Finite difference in space
ct = wildfire.fire(parameters)
W, B = ct.solvePDE('fd', 'rk4')
#%%
ct.plots(W, B)

#%%
# Chebyshev in space
ct = wildfire.fire(parameters)
Wc, Bc = ct.solvePDE('cheb', 'rk4')

#%%
ct.plots(Wc, Bc, True)

#%%
for i in range(T):
  if i % 10 == 0:
    ct.plotSimulation(i, W, True)
    ct.plotFuel(i, B, True)  
#%%
ct.save(W, B, 0)