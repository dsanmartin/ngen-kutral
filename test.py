import wildfire
import numpy as np
import matplotlib.pyplot as plt

# Helper to build reaction rate# Helpe 
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
  
def plotScalar(X, Y, U, title):
  plt.imshow(U(X,Y), origin="lower", cmap=plt.cm.jet, extent=[X[0,0], X[-1, -1], Y[0, 0], Y[-1, -1]])
  plt.title(title)
  plt.colorbar()
  plt.show()
  
  
# The resolution have to be lower than discrete version for computation of F
M, N = 50, 50
xa, xb = -1, 1
ya, yb = -1, 1
x = np.linspace(xa, xb, M)
y = np.linspace(ya, yb, N)
X, Y = np.meshgrid(x, y)
Xv, Yv = np.mgrid[xa:xb:complex(0, M // 4), ya:yb:complex(0, N // 4)]

T = 500
dt = 1e-3

T_env = 300
Ea = 83.68
A = 1e9
rho = 1e2
C = 1
k = 1


v1 = lambda x, y: np.cos(y)
v2 = lambda x, y: -np.cos(y)
V = (v1, v2)#vectorialField() #

a = lambda x, y: x*0 + 1# S(x+.25, y+.25) #x*0 + 1
#u0 = lambda x, y: 1e1*np.exp(-40*((x+.75)**2 + (y+.75)**2))
u0 = lambda x, y: 1e1*np.exp(-40*((x+.75)**2 + (y-.75)**2))

plotField(Xv, Yv, V)
plotScalar(X, Y, a, "Reaction")
plotScalar(X, Y, u0, "Initial contidion")

# Parameters
parameters = {
    'u0': u0,#initial,
    'beta0': a,
    'kappa': 0,#5e-2,
    'epsilon': 1e-1,
    'upc': .1,#np.random.rand(M, N),
    'q': 1e-1,#np.ones_like(initial)*.1,
    'v': V,
    'alpha': 1e-2,
    'x': np.linspace(xa, xb, M),
    'y': np.linspace(ya, yb, N),
    't': np.linspace(0, dt*T, T)
}
#%%
# We have to include border conditions, for now only 
# use dirichlet f(x,y) = u(x,y) for (x,y) \in \partial\Omega
ct = wildfire.fire(parameters)

W, B = ct.solvePDE(method='rk4')
#%%
ct.plots(W, B)#, True)
    
#%%
for i in range(T):
  if i % 10 == 0:
    ct.plotSimulation(i, W, True)
    ct.plotFuel(i, B, True)    
#%%
ct.save(W, B, 0)

