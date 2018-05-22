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

def temperatureFocus(M, N):
    x = np.linspace(xa, xb, N)
    y = np.linspace(ya, yb, M)
    X, Y = np.meshgrid(x, y)
    A = np.zeros((M,N))
    A[M//2,N//2] = 1.0
    A[M//2+1,N//2] = 1.0
    #A = S(X, Y)
    #A = A / np.max(A)
    return 1e3*np.exp(-40*((X-.5)**2 + (Y-.5)**2)), A

def vectorialField():
  # Vectorial field
  v1 = lambda x, y: (x*0) + 1
  v2 = lambda x, y: np.sin(x**2 + y**2)
  
  return (v1, v2)

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
M, N = 40, 40
xa, xb = -1, 1
ya, yb = -1, 1
x = np.linspace(xa, xb, M)
y = np.linspace(ya, yb, N)
X, Y = np.meshgrid(x, x)
Xv, Yv = np.mgrid[xa:xb:complex(0, M // 2), ya:yb:complex(0, N // 2)]

T = 100
dt = 1e-3#5

T_env = 300
Ea = 83.68
A = 1e9
rho = 1e2
C = 1
k = 1

# Initial conditions
initial, B = temperatureFocus(M, N)

v1 = lambda x, y: x #+ 1
v2 = lambda x, y: y #np.sin(x**2 + y**2)
V = (v1, v2)#vectorialField()

a = lambda x, y: 10*S(x, y)
u0 = lambda x, y: 1e3*np.exp(-40*((x-.0)**2 + (y-.0)**2))

plotField(Xv, Yv, V)
plotScalar(X, Y, a, "Reaction")
plotScalar(X, Y, u0, "Initial contidion")


# Parameters
parameters = {
    'u0': u0,#initial,
    'beta0': a,
    'kappa': 8e-1,
    'epsilon': .003,
    'upc': .1,#np.random.rand(M, N),
    'q': 1,#np.ones_like(initial)*.1,
    'v': V,
    'alpha': 1e-5,
    'x': np.linspace(xa, xb, M),
    'y': np.linspace(ya, yb, N),
    't': np.linspace(0, dt*T, T)
}
#%%
# We have to include border conditions, for now only 
# use dirichlet f(x,y) = u(x,y) for (x,y) \in \partial\Omega
ct = wildfire.fire(parameters)

W, B = ct.solvePDE(method='rk4')
#W = ct.solvePDECheb()

for i in range(T):
  if i % 10 == 0:
    ct.plotTemperatures(i, W)
#%%
ct = wildfire.fire(parameters)

Wc, _ = ct.solvePDE(method='cheb')

for i in range(T):
  if i % 10 == 0:
    ct.plotTemperaturesCheb(i, Wc)
#%%
for i in range(T):
  ct.plotTemperaturesCheb(i, Wc)