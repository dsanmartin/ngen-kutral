import wildfire
import numpy as np
import matplotlib.pyplot as plt

def temperatureFocus2(M, N):
    temperature = np.zeros((M,N))
    A = np.zeros((M,N))
    A[M//2,N//2] = 1.0
    A[M//2+1,N//2] = 1.0
    #A[M//2-1:M//2+1, N//2-1:N//2+1] = np.ones((4, 4))
    temperature = temperature + A * 100
    #A = np.zeros((M,N))
    return temperature, A

def temperatureFocus(M, N):
    x = np.linspace(0, 1, N)
    y = np.linspace(0, 1, M)
    X, Y = np.meshgrid(x, y)
    A = np.zeros((M,N))
    A[M//2,N//2] = 1.0
    A[M//2+1,N//2] = 1.0
    return 1e1*np.exp(-1000*((X-.5)**2 + (Y-.5)**2)), A

def vectorialField():
  # Vectorial field
  v1 = lambda x, y: (x*0) + 1
  v2 = lambda x, y: 10*np.sin(x**2 + y**2)
  
  return (v1, v2)
  

  
# The resolution have to be lower than discrete version for computation of F
M, N = 100, 100

T_env = 300
Ea = 83.68
A = 1e9
rho = 1e2
C = 1
k = 1

# Initial conditions
initial, B = temperatureFocus(M, N)
print(np.max(B))

V = vectorialField()

# Parameters
parameters = {
    'u0': initial,
    'beta0': B,
    'kappa': 1e-1,
    'epsilon': .003,
    'upc': .1,#np.random.rand(M, N),
    'q': 1,#np.ones_like(initial)*.1,
    'v': V,
    'alpha': 1e-3,
    'dt': 1e-4,
    'T': 1000
}
#%%
# We have to include border conditions, for now only 
# use dirichlet f(x,y) = u(x,y) for (x,y) \in \partial\Omega
ct = wildfire.fire(parameters)

W, B = ct.solvePDE()
#spde1 = ct.solveSPDE1(1/30)
#spde2 = ct.solveSPDE2(1/5)

for i in range(parameters['T']):
  if i % 100 == 0:
    ct.plotTemperatures(i, W)
#%%
    
## Discrete
#dtemp = temp.discrete(mu, initial, T, A, b, maxTemp)
#dtemps, _ = dtemp.propagate(4/30, 20)
#
#for i in range(T):
#  if i % 10 == 0:
#    dtemp.plotTemperatures(i, dtemps)