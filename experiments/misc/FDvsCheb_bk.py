# Comparing FD vs Cheb using Asensio et. al 2002
import numpy as np
from wildfire.fire import Fire
from wildfire import plots as p
import matplotlib.pyplot as plt
from wildfire.diffmat import chebyshevMatrix
#from scipy.interpolate import interp2d
#%%
# Asensio 2002 experiment
M, N = 128, 128
L = 500 # Timesteps
dt = 1e-10 # dt
#xa, xb = 0, 90 # x domain limit
#ya, yb = 0, 90 # y domain limit
xa, xb = -1, 1
ya, yb = -1, 1
x = np.linspace(xa, xb, N) # x domain
y = np.linspace(ya, yb, M) # y domain
t = np.linspace(0, dt*L, L) # t domain

# Domain transformations
xt = lambda t: (xb-xa)/2*t + (xb+xa)/2 # [-1,1] to [xa, xb]
tx = lambda x: 2/(xb-xa)*x-(xb+xa)/(xb-xa) # [xa, xb] to [-1, 1]

# Temperature initial condition
#u0 = lambda x, y: 6e0 * np.exp(-5e-2*((x-20)**2 + (y-20)**2)) 
#u0c = lambda x, y: tx(6e0) * np.exp(-tx(5e-2)*((tx(x)-tx(20))**2 + (tx(y)-tx(20))**2))
u0 = lambda x, y: 6 * np.exp(-2e2*((x+.65)**2 + (y+.65)**2))  
#u0 = lambda x, y: 1e1*np.exp(-150*((x+.7)**2 + (y-.7)**2)) 

# Fuel initial condition
np.random.seed(666)
#b0 = lambda x, y: np.round(np.random.uniform(size=(x.shape)), decimals=2)
b0 = lambda x, y: x*0 + 1

# Wind effect
gamma = 2
v1 = lambda x, y, t: gamma * np.cos(np.pi/4 + x*0 + 1e-2*t) # 300
v2 = lambda x, y, t: gamma * np.sin(np.pi/4 + x*0 + 1e-2*t) # 300
V = (v1, v2)

#w1 = lambda x, y, t: gamma * np.cos((7/4+.01*t+x*0)*np.pi)
#w2 = lambda x, y, t: gamma * np.sin((7/4+.01*t+y*0)*np.pi)
#W = (w1, w2)

# Parameters
#kappa = 1e-1 # diffusion coefficient
#epsilon = 3e-1#3e-2 # inverse of activation energy
#upc = 3e0 # u phase change
#q = 1 # reaction heat
#alpha = 1e-3 # natural convection

kappa = 1e-2#5e-3 # diffusion coefficient
epsilon = 1e-1#1e-1 # inverse of activation energy
upc = 3#1e-1 # u phase change
q = 1e1#1e-1 # reaction heat
alpha = 1e-3#1e1#1e-1 # natural convection
#%%
#X, Y = np.meshgrid(x, y)
#plt.imshow(u0(X, Y), origin="lower")
#plt.colorbar()
#%%
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
    'sparse': False,
    'show': False,
    'complete': False
}

ct = Fire(parameters)
#%%
"""
#parameters['u0'] = u0c
Uc, Bc = ct.solvePDE('cheb', 'last')
#parameters['u0'] = u0
#Ufd, Bfd = ct.solvePDE('fd', 'last')
#%%
#import matplotlib.pyplot as plt
##plt.imshow(Ufd, origin="lower",cmap=plt.cm.jet)
##for i in range(L):
##  if i % 10 != 0: continue
#plt.imshow(Ufd, origin="lower", cmap=plt.cm.jet, vmin=np.min(Ufd), vmax=np.max(Ufd))
#plt.colorbar()
#plt.show()
np.save('experiments/misc/fdvscheb/1e-10/fd/U128', Ufd)
#%%
_, xc = chebyshevMatrix(N-1)
_, yc = chebyshevMatrix(M-1)
fu = interp2d(xc, yc, Uc, kind='linear')          
fb = interp2d(xc, yc, Bc, kind='linear')
#plt.imshow(fu(x, y), origin="lower")
Xc, Yc = np.meshgrid(xc, yc)
#plt.contourf(Xc, Yc, Uc, origin="lower", cmap=plt.cm.jet)
Uc = fu(x, y)
plt.imshow(Uc, origin="lower", cmap=plt.cm.jet)
plt.colorbar()
#%%
#np.save('experiments/misc/fdvscheb/fd/U128', Ufd)

#%%
#Uc = fu(x, y)
np.save('experiments/misc/fdvscheb/1e-10/cheb/U1024', Uc)
"""
#%%
#times
meanC = np.array([300/1000, 1.21, 1.13, 4.16, 23.6])
meanF = np.array([296/1000, 1, 1.05, 3.97, 21.8])
stdC = np.array([31.6, 88, 60.9, 295, 23.9*1000]) / 1000
stdF = np.array([33.5, 72.3, 57.7, 82.8, 256*100]) / 1000
#%%
Usol = np.load("experiments/misc/fdvscheb/1e-8/Usol.npy")
Uc256 = np.load("experiments/misc/fdvscheb/1e-8/cheb/U256.npy")
Uc128 = np.load("experiments/misc/fdvscheb/1e-8/cheb/U128.npy")
Uc64 = np.load("experiments/misc/fdvscheb/1e-8/cheb/U64.npy")
Uc32 = np.load("experiments/misc/fdvscheb/1e-8/cheb/U32.npy")
Uc16 = np.load("experiments/misc/fdvscheb/1e-8/cheb/U16.npy")
Uf256 = np.load("experiments/misc/fdvscheb/1e-8/fd/U256.npy")
Uf128 = np.load("experiments/misc/fdvscheb/1e-8/fd/U128.npy")
Uf64 = np.load("experiments/misc/fdvscheb/1e-8/fd/U64.npy")
Uf32 = np.load("experiments/misc/fdvscheb/1e-8/fd/U32.npy")
Uf16 = np.load("experiments/misc/fdvscheb/1e-8/fd/U16.npy")

cheb_err = np.array([
    np.linalg.norm((Usol[::32,::32] - Uc16).flatten(), np.inf),
    np.linalg.norm((Usol[::16,::16] - Uc32).flatten(), np.inf),
    np.linalg.norm((Usol[::8,::8] - Uc64).flatten(), np.inf),
    np.linalg.norm((Usol[::4,::4] - Uc128).flatten(), np.inf),
    np.linalg.norm((Usol[::2,::2] - Uc256).flatten(), np.inf),
])

fd_err = np.array([
    np.linalg.norm((Usol[::32,::32] - Uf16).flatten(), np.inf),
    np.linalg.norm((Usol[::16,::16] - Uf32).flatten(), np.inf),
    np.linalg.norm((Usol[::8,::8] - Uf64).flatten(), np.inf),
    np.linalg.norm((Usol[::4,::4] - Uf128).flatten(), np.inf),
    np.linalg.norm((Usol[::2,::2] - Uf256).flatten(), np.inf),
])
#%%


#%%
plt.plot(meanC, cheb_err_2, "b-o", label="Chebyshev")
plt.plot(meanF, fd_err, "r-x", label="FD")
plt.xlabel("t")
plt.ylabel("Error")
plt.grid(True)
plt.legend()
plt.show()
#plt.imshow(Uc64, origin="lower", cmap=plt.cm.jet)
#plt.colorbar()
#%%
import numpy as np
import matplotlib.pyplot as plt
N = np.array([2**i for i in range(4, 9)])
plt.plot(N, cheb_err_2, 'b-o')
plt.plot(N, fd_err, 'r-x')
plt.grid(True)
plt.xscale('log')
plt.yscale('log')
plt.show()



#%%%
#%%
Usol = np.load("experiments/misc/fdvscheb/1e-10/Usol.npy")
Uc1024 = np.load("experiments/misc/fdvscheb/1e-10/cheb/U1024.npy")
Uc512 = np.load("experiments/misc/fdvscheb/1e-10/cheb/U512.npy")
Uc256 = np.load("experiments/misc/fdvscheb/1e-10/cheb/U256.npy")
Uc128 = np.load("experiments/misc/fdvscheb/1e-10/cheb/U128.npy")
#Uc64 = np.load("experiments/misc/fdvscheb/1e-10/cheb/U64.npy")
#Uc32 = np.load("experiments/misc/fdvscheb/1e-10/cheb/U32.npy")
#Uc16 = np.load("experiments/misc/fdvscheb/1e-10/cheb/U16.npy")
Uf1024 = np.load("experiments/misc/fdvscheb/1e-10/fd/U1024.npy")
Uf512 = np.load("experiments/misc/fdvscheb/1e-10/fd/U512.npy")
Uf256 = np.load("experiments/misc/fdvscheb/1e-10/fd/U256.npy")
Uf128 = np.load("experiments/misc/fdvscheb/1e-10/fd/U128.npy")
#Uf64 = np.load("experiments/misc/fdvscheb/1e-8/fd/U64.npy")
#Uf32 = np.load("experiments/misc/fdvscheb/1e-8/fd/U32.npy")
#Uf16 = np.load("experiments/misc/fdvscheb/1e-8/fd/U16.npy")

"""
cheb_err = np.array([
#    np.linalg.norm((Usol[::32,::32] - Uc16).flatten(), np.inf),
#    np.linalg.norm((Usol[::16,::16] - Uc32).flatten(), np.inf),
#    np.linalg.norm((Usol[::8,::8] - Uc64).flatten(), np.inf),
    np.linalg.norm((Usol[::16,::16] - Uc128).flatten(), np.inf),
    np.linalg.norm((Usol[::8,::8] - Uc256).flatten(), np.inf),
    np.linalg.norm((Usol[::4,::4] - Uc512).flatten(), np.inf),
    np.linalg.norm((Usol[::2,::2] - Uc1024).flatten(), np.inf),
])

fd_err = np.array([
#    np.linalg.norm((Usol[::32,::32] - Uf16).flatten(), np.inf),
    np.linalg.norm((Usol[::16,::16] - Uf128).flatten(), np.inf),
    np.linalg.norm((Usol[::8,::8] - Uf256).flatten(), np.inf),
    np.linalg.norm((Usol[::4,::4] - Uf512).flatten(), np.inf),
    np.linalg.norm((Usol[::2,::2] - Uf1024).flatten(), np.inf),
])
"""
#%%%
from bary2d import *

interpolations = list()
Ucs = [Uc128, Uc256, Uc512]
nodes = np.array([128, 256, 512])
_, xe = chebyshevMatrix(1023)
_, ye = chebyshevMatrix(1023)

for i in range(len(Ucs)):
  wi = weights(nodes[i])
  wj = weights(nodes[i])

  _, xc = chebyshevMatrix(nodes[i]-1)
  _, yc = chebyshevMatrix(nodes[i]-1)
  
  inter = interpolation2D(xe, ye, xc, yc, Ucs[i], wi, wj, BL2Dnp)
  interpolations.append(inter)
#%%
cheb_err_2 = np.array([
    np.linalg.norm((Uc1024 - interpolations[0]).flatten(), np.inf),
    np.linalg.norm((Uc1024- interpolations[1]).flatten(), np.inf),
    np.linalg.norm((Uc1024 - interpolations[2]).flatten(), np.inf)
])

fd_err_2 = np.array([
#    np.linalg.norm((Usol[::32,::32] - Uf16).flatten(), np.inf),
    np.linalg.norm((Uf1024[::8,::8] - Uf128).flatten(), np.inf),
    np.linalg.norm((Uf1024[::4,::4] - Uf256).flatten(), np.inf),
    np.linalg.norm((Uf1024[::2,::2] - Uf512).flatten(), np.inf),
    #np.linalg.norm((Usol[::2,::2] - Uf1024).flatten(), np.inf),
])
#%%
import matplotlib.pyplot as plt

N = np.array([2**i for i in range(7, 10)])
plt.plot(N, cheb_err_2, 'b-o', label="Cheb")
plt.plot(N, fd_err_2, 'r-x', label="FD")
#plt.xscale('log')
plt.yscale('log')
plt.xlabel("N")
plt.ylabel("Error")
plt.xticks(N, map(str, N))
plt.grid(True)
plt.legend()
plt.show()