#%%
import numpy as np
from wildfire import Fire
from wildfire.utils import old_plots as p
import time
#%%
x0 = -20
y0 = -20

eps = -5e-2#-5e-2

# Temperature initial condition
u0 = lambda x, y: 7e0*np.exp(eps*((x+x0)**2 + (y+y0)**2)) #+ \
   #7e0*np.exp(-5e-2*((x+x0-6)**2 + (y+y0)**2)) + \
   #8e0*np.exp(-5e-2*((x+x0)**2 + (y+y0-6)**2))

# Fuel initial condition
b0 = lambda x, y: x * 0 + 1
#def b0(x, y):
#  B0 = b0_in(x, y)
#  M, N = B0.shape
#  B0[0,:] = np.zeros(N); B0[-1,:] = np.zeros(N)
#  B0[:,0] = np.zeros(M); B0[:,-1] = np.zeros(M)
#  return B0
  

# Wind effect
gamma = 1
v1 = lambda x, y, t: gamma * np.cos(np.pi/4 + x * 0 - t * 0)#.025) 
v2 = lambda x, y, t: gamma * np.sin(np.pi/4 + x * 0 - t * 0)#.025) 
V = lambda x, y, t: (v1(x, y, t), v2(x, y, t)) #V = (v1, v2)

e = 9
Nx = 2 ** e
Ny = 2 ** e
Nt = 1

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
    't_lim': (0, 1e-10),
}

ct = Fire(**parameters)

#%%
_, _, _, Ufd, Bfd = ct.solvePDE(Nx, Ny, Ny, u0, b0, V)
#%%
_, _, _, Ufft, Bfft = ct.solvePDE(Nx, Ny, Ny, u0, b0, V, space_method='fft')
#%%
p.plotCompare((0, 90), (0, 90), Ufd, Bfd, Ufft, Bfft, "FD 2", "FFT")
#%%
fd_times = []
fft_times = []
fd_errs = []
fft_errs = []
exps = np.arange(4, e)

for exp in exps:
    Nx, Ny = 2 ** exp, 2 ** exp
    print("Nx: %d, Ny: %d" % (Nx, Ny))

    start = time.time()
    _, _, _, Ufd_e, Bfd_e = ct.solvePDE(Nx, Ny, Nt, u0, b0, V)
    end = time.time()
    fd_times.append(end - start)

    # FFT
    start = time.time()
    _, _, _, Ufft_e, Bfft2_e = ct.solvePDE(Nx, Ny, Nt, u0, b0, V, space_method='FFT')
    end = time.time()
    fft_times.append(end - start)

    step = 2**(e - exp)
    fd_err = np.linalg.norm((Ufd[::step,::step] - Ufd_e).flatten(), np.inf) #/ np.linalg.norm(Ufd[::step,::step].flatten(), np.inf)
    fft_err = np.linalg.norm((Ufft[::step,::step] - Ufft_e).flatten(), np.inf) #/ np.linalg.norm(Ufft[::step,::step].flatten(), np.inf)
    fd_errs.append(fd_err)
    fft_errs.append(fft_err)

#%%
#sim_id, dir_base = utils.simulation()
#utils.saveParameters(dir_base, parameters)
x_min, x_max = parameters['x_lim']
N = np.array([2**i for i in exps])
times = np.zeros((len(N), 3))
errors = np.zeros((len(N), 3))
times[:,0] = N
times[:,1] = np.array(fd_times)#[fd.average for fd in fd_times])
times[:,2] = np.array(fft_times)#[fft.average for fft in fft_times])
errors[:,0] = (x_max - x_min) / (N - 1)
errors[:,1] = np.array(fd_errs)
errors[:,2] = np.array(fft_errs)
#utils.saveTimeError(dir_base, times, errors)
#%%
p.spatialComplexity(times)
#%%
p.spatialConvergence(errors, c=40)