import numpy as np
import argparse
import wildfire
from wildfire.utils.functions import G
from wildfire.utils import plots
from wildfire.utils import storage

parser = argparse.ArgumentParser(description='Create and execute a wildfire simulation.')

# Model parameters
parser.add_argument('-kap', '--kappa', metavar='K', default=1e-1, type=float, 
	help='kappa parameter (default: 1e-1)')
parser.add_argument('-eps', '--epsilon', metavar='E', default=3e-1, type=float, 
	help='epsilon parameter (default: 3e-1)')
parser.add_argument('-upc', '--phase', metavar='P', default=3, type=float,
	help='phase change threshold parameter (default: 3)')
parser.add_argument('-alp', '--alpha', metavar='A', default=1e-3, type=float, 
	help='alpha parameter (default: 1e-3)')
parser.add_argument('-qrh', '--reaction', metavar='Q', default=1, type=float, 
	help='alpha parameter (default: 1.0)')
parser.add_argument('-xlim', '--xlimits', metavar=('x_min', 'x_max'), default=(0, 90), type=float, nargs=2, 
  help='x domain limits (default: [0, 90])')
parser.add_argument('-ylim', '--ylimits', metavar=('y_min', 'y_max'), default=(0, 90), type=float, nargs=2, 
  help='y domain limits (default: [0, 90])')
parser.add_argument('-tlim', '--tlimits', metavar=('t_min', 't_max'), default=(0, 30), type=float, nargs=2, 
  help='t domain limits (default: [0, 30])')

# Numerical parameters
parser.add_argument('-sm', '--space', metavar='SM', type=str, help='Space method approximation', required=True)
parser.add_argument('-Nx', '--xnodes', metavar='Nx', type=int, help='Number of nodes in x', required=True)
parser.add_argument('-Ny', '--ynodes', metavar='Ny', type=int, help='Number of nodes in y', required=True)
parser.add_argument('-tm', '--time', metavar='TM', type=str, help='Time method approximation', required=True)
parser.add_argument('-Nt', '--tnodes', metavar='Nt', type=int, help='Number of nodes in t', required=True)

# Initial conditions 
parser.add_argument('-u0', '--initial-temperature', metavar='U0', type=str, 
	help='Initial temperature file. Only .csv and .npy supported.', required=True)
parser.add_argument('-b0', '--initial-fuel', metavar='B0', type=str, 
	help='Initial fuel file. Only .csv and .npy supported.', required=True)

# Others parameters
parser.add_argument('-acc', '--accuracy', metavar='ACC', default=2, type=int, 
	help='Finite difference accuracy (default: 2)')
parser.add_argument('-sps', '--sparse', metavar='S', default=False, type=bool, 
	help='Finite difference sparse matrices (default: False)')
parser.add_argument('-lst', '--last', metavar='LST', default=True, type=bool, 
	help='Only last approximation (default: True)')
parser.add_argument('-plt', '--plot', metavar='PLT', default=False, type=bool,
	help='Plot result (default: False)')

args = parser.parse_args()
#print(args)

### PARAMETERS ###
# Model parameters #
kap = args.kappa
eps = args.epsilon
upc = args.phase
alp = args.alpha
q = args.reaction
x_min, x_max = args.xlimits
y_min, y_max = args.ylimits
t_min, t_max = args.tlimits

# Numerical #
# Space
space_method = args.space
Nx = args.xnodes
Ny = args.ynodes
acc = args.accuracy
sparse = args.sparse

# Time
time_method = args.time
Nt = args.tnodes
last = args.last

# Initial conditions
u0_dir = args.initial_temperature
b0_dir = args.initial_fuel

# Temperature
u0 = storage.openFile(u0_dir)
# if '.npy' in u0_dir:
# 	u0 = np.load(u0_dir)
# elif '.csv' in u0_dir:
# 	u0 = np.loadtxt(u0_dir)
# else:
# 	raise Exception("File extension not supported.")

# Fuel
b0 = storage.openFile(b0_dir)
# if '.npy' in b0_dir:
# 	b0 = np.load(b0_dir)
# elif '.csv' in b0_dir:
# 	b0 = np.loadtxt(b0_dir)
# else:
# 	raise Exception("File extension not supported.")

# Wind effect
gamma = 1
w1 = lambda x, y, t: gamma * np.cos(np.pi/4 + x * 0)
w2 = lambda x, y, t: gamma * np.sin(np.pi/4 + x * 0)
V = (w1, w2)
### PARAMETERS ###

# Physical parameters for the model
physical_parameters = {    
    'kap': kap, # diffusion coefficient
    'eps': eps, # inverse of activation energy
    'upc': upc, # u phase change
    'q': q, # reaction heat
    'alp': alp, # natural convection,
    # Domain
    'x_lim': (x_min, x_max), # x-axis domain 
    'y_lim': (y_min, y_max), # y-axis domain
    't_lim': (t_min, t_max) # time domain
}

wildfire_ = wildfire.Fire(**physical_parameters)
t, X, Y, U, B = wildfire_.solvePDE(Nx, Ny, Nt, u0, b0, V, space_method, time_method, last=last, acc=acc, sparse=sparse)

if args.plot and args.last:
	plots.UB(t, X, Y, U, B, V)