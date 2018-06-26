import wildfire
import numpy as np
import matplotlib.pyplot as plt
#%%
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
  plt.quiver(Xv, Yv, V[0](Xv, Yv, 0), V[1](Xv, Yv, 0))  
  plt.title("Wind")
  plt.show()
  
def plotScalar(X, Y, U, title, cmap_):
  plt.imshow(U(X,Y), origin="lower", cmap=cmap_, 
             extent=[X[0,0], X[-1, -1], Y[0, 0], Y[-1, -1]])
  plt.title(title)
  plt.colorbar()
  plt.show()
  
# Domain: [-1, 1]^2 x [0, T*dt]
M, N = 64, 64 # Resolution
T = 1000 # Timesteps
dt = 1e-3 # dt
xa, xb = -1, 1 # x domain limit
ya, yb = -1, 1 # y domain limit
x = np.linspace(xa, xb, N) # x domain
y = np.linspace(ya, yb, M) # y domain
t = np.linspace(0, dt*T, T) # t domain

# Meshes for initial condition plots
X, Y = np.meshgrid(x, y)
Xv, Yv = np.mgrid[xa:xb:complex(0, N // np.sqrt(N)), ya:yb:complex(0, M // np.sqrt(M))]

# TODO: define parameters in function of real conditions
#T_env = 300
#Ea = 83.68
#A = 1e9
#rho = 1e2
#C = 1
#k = 1

# Vector field V = (v1, v2). "Incompressible flow div(V) = 0"
gamma = 1
v1 = lambda x, y, t: gamma * np.cos((7/4+.5*t)*np.pi) 
v2 = lambda x, y, t: gamma * np.sin((7/4+.5*t)*np.pi)
V = (v1, v2)

# Lambda function for temperature initial condition
u0 = lambda x, y: 1e1*np.exp(-150*((x+.5)**2 + (y-.25)**2)) 

# Lambda function for fuel initial condition
b0 = lambda x, y: x*0 + 1 #S(x+.25, y+.25) #x*0 + 1

# Non dimensional parameters
kappa = 5e-3 # diffusion coefficient
epsilon = 1*1e-1 # inverse of activation energy
upc = 1*.1 # u phase change
q = 1*1e-1 # reaction heat
alpha = 0#1e-1 # natural convection

# Plot initial conditions
plotField(Xv, Yv, V)
plotScalar(X, Y, b0, "Fuel", plt.cm.Oranges)
plotScalar(X, Y, u0, "Initial condition", plt.cm.jet)

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

ct = wildfire.fire(parameters)
#%%
# Finite difference in space
W, B = ct.solvePDE('fd', 'rk4')
#%%
ct.plots(W, B)

#%%
# Chebyshev in space
Wc, Bc = ct.solvePDE('cheb', 'rk4')
#%%
ct.plots(Wc, Bc, True)

#%%
W_1024 = np.load('data/last_W_1024.npy')
W_512 = np.load('data/last_W_512.npy')
W_256 = np.load('data/last_W_256.npy')
W_128 = np.load('data/last_W_128.npy')
#%%
#errors = np.array([
#    np.max(np.abs(W_128 - W_1024[::8, ::8])),
#    np.max(np.abs(W_256 - W_1024[::4, ::4])),
#    np.max(np.abs(W_512 - W_1024[::2, ::2]))
#    ])
errors = np.array([
    np.linalg.norm((W_128 - W_1024[::8, ::8]).flatten(), np.inf),
    np.linalg.norm((W_256 - W_1024[::4, ::4]).flatten(), np.inf),
    np.linalg.norm((W_512 - W_1024[::2, ::2]).flatten(), np.inf),
    ])
h = np.array([2/(2**i) for i in range(7, 10)])
#%%
plt.plot(h, errors, '-x')
plt.xlabel("h")
plt.grid(True)
plt.show()

