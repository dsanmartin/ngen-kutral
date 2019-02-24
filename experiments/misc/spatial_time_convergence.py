import numpy as np
import pathlib
import sys
from wildfire.fire import Fire

if len(sys.argv) < 4:
  print("Arguments error")
  sys.exit()

BASE_DIR = sys.argv[1]#"/user/d/dsanmart/wildfires/ngen-kutral/experiments/misc/spatial_time_convergence/"
N = int(sys.argv[2])
L = int(sys.argv[3])

# Parameters for the model (to compare with chebyshev)
#parameters = {
#    'u0': lambda x, y: 6 * np.exp(-2e2*((x+.65)**2 + (y+.65)**2)), 
#    'beta0': lambda x, y: x*0 + 1,
#    'v': (lambda x, y, t: 2 * np.cos(np.pi/4 + x*0), 
#          lambda x, y, t: 2 * np.sin(np.pi/4 + x*0)),
#    'kappa': 1e-2, 
#    'epsilon': 1e-1, 
#    'upc': 3, 
#    'q': 1e1, 
#    'alpha': 1e-3,
#    'sparse': False,
#    'show': False,
#    'complete': False,
#    'components': (True, True, False) # Remove f(u,b)
#}
# For finite difference test
parameters = {
    'u0': lambda x, y: 6e0 * np.exp(-5e-2*((x-20)**2 + (y-20)**2)), 
    'beta0': lambda x, y: x*0 + 1,
    'v': (lambda x, y, t: 1 * np.cos(np.pi/4 + x*0),
          lambda x, y, t: 1 * np.sin(np.pi/4 + x*0)),
    'kappa': 1e-1, 
    'epsilon': 3e-1, 
    'upc': 3, 
    'q': 1, 
    'alpha': 1e-3, 
    'sparse': True,
    'show': False,
    'complete': False,
    'components': (True, True, False)
}

dir_name = BASE_DIR + "/" + sys.argv[3] + "/fd/"

pathlib.Path(dir_name).mkdir(parents=True, exist_ok=True) 

parameters['x'] = np.linspace(0, 90, N)
parameters['y'] = np.linspace(0, 90, N)
parameters['t'] = np.linspace(0, 1, L)    

ct = Fire(parameters)
# Solve
# Uc, Bc = ct.solvePDE('cheb', 'last')
Ufd, Bfd = ct.solvePDE('fd', 'veclast')
np.save(dir_name + "U" + str(N), Ufd)
