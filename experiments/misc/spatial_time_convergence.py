import numpy as np
import pathlib
from wildfire.fire import Fire
from wildfire import plots as p
import matplotlib.pyplot as plt
from wildfire.diffmat import chebyshevMatrix
#%%
BASE_DIR = "experiments/misc/spatial_time_convergence/"

# Parameters for the model
parameters = {
    'u0': lambda x, y: 6 * np.exp(-2e2*((x+.65)**2 + (y+.65)**2)), 
    'beta0': lambda x, y: x*0 + 1,
    'v': (lambda x, y, t: 2 * np.cos(np.pi/4 + x*0 + 1e-2*t), 
          lambda x, y, t: 2 * np.sin(np.pi/4 + x*0 + 1e-2*t)),
    'kappa': 1e-2, 
    'epsilon': 1e-1, 
    'upc': 3, 
    'q': 1e1, 
    'alpha': 1e-3,
    'sparse': False,
    'show': False,
    'complete': False
}
#%%
Ns = np.array([2 ** j for j in range(4, 11)])
Ts = np.array([i for i in range(4, 12, 2)])
TsN = np.array([10**i+1 for i in range(4, 12, 2)])
ii = 0
for ts in Ts:
  for N in Ns:
    
    dt_str = '1e-' + str(ts)
    dt = float(dt_str)
    print("Simulation using: dt = {0}, N = {1}".format(dt, N))
    print(TsN[ii])
    
    dir_name = BASE_DIR + dt_str + "/fd/"

    pathlib.Path(dir_name).mkdir(parents=True, exist_ok=True) 
    
    parameters['x'] = np.linspace(-1, 1, N)
    parameters['y'] = np.linspace(-1, 1, N)
    parameters['t'] = np.linspace(0, 1, TsN[ii])    
    
    ct = Fire(parameters)
    # Solve
    # Uc, Bc = ct.solvePDE('cheb', 'last')
    Ufd, Bfd = ct.solvePDE('fd', 'last')
    np.save(dir_name + "U" + str(N), Ufd)
  ii += 1
#%%

Ns = np.array([2 ** j for j in range(4, 11)])
Ts = np.array([i for i in range(4, 12, 2)])

errors_ = np.zeros((len(Ts), len(Ns) - 1))

for i in range(len(Ts)):
  U1024 = np.load('experiments/misc/spatial_time_convergence/1e-' + str(Ts[i]) + '/fd/U1024.npy')
  for j in range(len(Ns)-1):
    Utmp = np.load('experiments/misc/spatial_time_convergence/1e-' + str(Ts[i]) + '/fd/U' + str(Ns[j]) + '.npy')
    errors_[i, j] = np.linalg.norm((U1024[::1024 // Ns[j],:: 1024 // Ns[j]] - Utmp).flatten(), np.inf)
#%%   
plt.figure(figsize=(8, 6))
plot = plt.imshow(errors_[::,::-1], cmap=plt.cm.coolwarm)
plt.xticks(np.arange(len(Ns)-1), map(lambda N: '2/' + str(N) , Ns[-2::-1]))
plt.yticks(np.arange(len(Ts)), map(lambda T: '1e-' + str(T), Ts))
cbar = plt.colorbar(plot, fraction=0.046, pad=0.04)
cbar.set_label(r"$||U_{ref} - U_{N}||_{\infty}$", fontsize=14)
plt.xlabel(r"$h$", fontsize=14)
plt.ylabel(r"$\Delta t$", fontsize=16)
plt.tick_params(axis='both', which='major', labelsize=12)
plt.savefig('st_conv.pdf', format='pdf')
plt.show()
#%%
plt.plot(Ns[:-1], errors_[0])