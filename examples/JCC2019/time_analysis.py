#%%
import numpy as np
from wildfire import Fire
from wildfire.utils import old_plots as p
import time
#%%
x0 = -20
y0 = -20

# Temperature initial condition
u0 = lambda x, y: 7e0*np.exp(-5e-2*((x+x0)**2 + (y+y0)**2))

# Fuel initial condition
b0 = lambda x, y: x * 0 + 1

# Wind effect
gamma = 1
v1 = lambda x, y, t: gamma * np.cos(np.pi/4 + x * 0 - t * 0)#.025) 
v2 = lambda x, y, t: gamma * np.sin(np.pi/4 + x * 0 - t * 0)#.025) 
V = (v1, v2)

e = 12
Nt = 2 ** e
Nx = 2 ** 7
Ny = 2 ** 7

# Parameters for the model
parameters = {
  # Physical
  'kap': 1e-1, # diffusion coefficient
  'eps': 3e-1, #3e-2 # inverse of activation energy
  'upc': 3, # u phase change
  'q': 1, # reaction heat
  'alp': 1e-3, # natural convection
  'x_lim': (0, 90), 
  'y_lim': (0, 90), 
  't_lim': (0, 10),
}

ct = Fire(**parameters)

#%%
_, _, _, Ueul, Beul = ct.solvePDE(Nx, Ny, Nt, u0, b0, V, time_method='Euler')
#%%
_, _, _, Urk4, Brk4 = ct.solvePDE(Nx, Ny, Nt, u0, b0, V)
#%%
p.plotCompare((0, 90), (0, 90), Ueul, Beul, Urk4, Brk4, "Euler", "RK4")
#%%
#max_e = 10
#eul_times = []
#rk4_times = []
#eul_errs = []
#rk4_errs = []

exps = np.arange(6, e)

rep = 10

eul_times = np.zeros((len(exps), rep))
rk4_times = np.zeros((len(exps), rep))
eul_errs_U = np.zeros(len(exps))
rk4_errs_U = np.zeros(len(exps))
eul_errs_B = np.zeros(len(exps))
rk4_errs_B = np.zeros(len(exps))

#exps = np.arange(6, max_e)
#%%
#for e in exps:
#  parameters['L']= 2 ** e
#  print("L: %d" % (2 ** e))
#  
#  parameters['time'] = "Euler"
#  eul_time = %timeit -o Fire(parameters).solvePDE(keep="last")
#  Ueul_e, Beul_e = Fire(parameters).solvePDE(keep="last")
#  eul_times.append(eul_time)
#  
#  # FFT
#  parameters['time'] = "RK4"
#  rk4_time = %timeit -o Fire(parameters).solvePDE(keep="last")
#  Urk4_e, Brk4_e = Fire(parameters).solvePDE(keep="last")
#  rk4_times.append(rk4_time)
#  
#  #step = 2**(max_e - e)
#  eul_err = np.linalg.norm((Ueul - Ueul_e).flatten(), np.inf) #/ np.linalg.norm(Ufd.flatten(), np.inf)
#  rk4_err = np.linalg.norm((Urk4 - Urk4_e).flatten(), np.inf) #/ np.linalg.norm(Ufft[::step,::step].flatten(), np.inf)
#  eul_errs.append(eul_err)
#  rk4_errs.append(rk4_err)
  
for i in range(len(exps)):
  exp = exps[i]
  Nt = 2 ** exp
  print("Nt: %d" % (Nt))
  
  for r in range(rep):
    start = time.time()
    _, _, _, Ueul_e, Beul_e = ct.solvePDE(Nx, Ny, Nt, u0, b0, V,time_method='Euler')
    end = time.time()
    eul_times[i, r] = end - start
    
    # FFT
    start = time.time()
    _, _, _, Urk4_e, Brk4_e = ct.solvePDE(Nx, Ny, Nt, u0, b0, V)
    end = time.time()
    rk4_times[i, r] = end - start
  
  step = 2**(e - exp)
  eul_err_U = np.linalg.norm((Ueul - Ueul_e).flatten(), np.inf) / np.linalg.norm(Ueul.flatten(), np.inf)
  eul_err_B = np.linalg.norm((Beul - Beul_e).flatten(), np.inf) / np.linalg.norm(Beul.flatten(), np.inf)
  #fd_errs_U.append(fd_err_U)
  #fd_errs_B.append(fd_err_U)
  eul_errs_U[i] = eul_err_U
  eul_errs_B[i] = eul_err_B
  
  rk4_err_U = np.linalg.norm((Urk4 - Urk4_e).flatten(), np.inf) / np.linalg.norm(Urk4.flatten(), np.inf)
  rk4_err_B = np.linalg.norm((Brk4 - Brk4_e).flatten(), np.inf) / np.linalg.norm(Brk4.flatten(), np.inf)
  #fft_errs_U.append(fft_err_U)
  #fft_errs_B.append(fft_err_B)
  rk4_errs_U[i] = rk4_err_U
  rk4_errs_B[i] = rk4_err_B
#%%
#import matplotlib.pyplot as plt
#Ls = np.array([2**i for i in exps])
#eul_times_ = np.array([eul.average for eul in eul_times])
#rk4_times_ = np.array([rk4.average for rk4 in rk4_times])
#%%
#sim_id, dir_base = utils.simulation()
#utils.saveParameters(dir_base, parameters)
t_min, t_max = 0, 10
L = np.array([2**i for i in exps])
times = np.zeros((len(L), 3))
errors = np.zeros((len(L), 5))
times[:,0] = L
times[:,1] = np.mean(eul_times, axis=1) #eul_times_#np.array([eul.average for eul in eul_times]) #np.array(fd_times)#[fd.average for fd in fd_times])
times[:,2] = np.mean(rk4_times, axis=1) #rk4_times_#np.array([rk4.average for rk4 in rk4_times]) #np.array(fft_times)#[fft.average for fft in fft_times])
errors[:,0] = (t_max - t_min) / (L+1)
errors[:,1] = eul_errs_U
errors[:,2] = rk4_errs_U
errors[:,3] = eul_errs_B
errors[:,4] = rk4_errs_B
#utils.saveTimeError(dir_base, times, errors)

#%% COMPLEXITY
p.timeComplexity(times)
#%% CONVERGENCE
p.timeConvergence(errors)
# #%%
# for i in range(len(L)):
#   print("%d %.8f" % (L[i], times[i,2]))