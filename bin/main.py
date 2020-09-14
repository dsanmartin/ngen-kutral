import os
import numpy as np
import argparse
import wildfire
from wildfire.utils.functions import G
from wildfire.utils import plots
from wildfire.utils import storage

def b0_bc(B):
    rows, cols = B.shape
    B[ 0,:] = np.zeros(cols)
    B[-1,:] = np.zeros(cols)
    B[:, 0] = np.zeros(rows)
    B[:,-1] = np.zeros(rows)
    return B

def b0_(x, y):
    np.random.seed(777)
    br = lambda x, y: np.round(np.random.uniform(size=(x.shape)), decimals=2)
    return b0_bc(br(x, y))

parser = argparse.ArgumentParser(add_help=False, description='Create and execute a wildfire simulation.')

optional = parser._action_groups.pop() # Edited this line
required = parser.add_argument_group('required arguments')

#parser._action_groups.pop()
#required = parser.add_argument_group('required arguments')
#optional = parser.add_argument_group('optional arguments')

### REQUIRED PARAMETERS ###

# Numerical parameters
required.add_argument('-sm', '--space', metavar='SM', type=str, 
    help='Space method approximation, FD (Finite Difference) or FFT (Fast Fourier Transform).', required=True)
required.add_argument('-Nx', '--xnodes', metavar='NX', type=int, help='Number of nodes in x.', required=True)
required.add_argument('-Ny', '--ynodes', metavar='NY', type=int, help='Number of nodes in y.', required=True)
required.add_argument('-tm', '--time', metavar='TM', type=str, help='Time method approximation, Euler or RK4.', required=True)
required.add_argument('-Nt', '--tnodes', metavar='NT', type=int, help='Number of nodes in t.', required=True)

### OPTIONAL PARAMTERS ###

# Model parameters
parser.add_argument('-h', '--help', action='help', default=argparse.SUPPRESS,
    help='Show this help message and exit.')
parser.add_argument('-k', '--kappa', metavar='K', default=1e-1, type=float, 
    help='Kappa parameter (default: 0.1).')
parser.add_argument('-e', '--epsilon', metavar='E', default=3e-1, type=float, 
    help='Epsilon parameter (default: 0.3).')
parser.add_argument('-p', '--phase', metavar='P', default=3.0, type=float,
    help='Phase change threshold parameter (default: 3.0).')
parser.add_argument('-a', '--alpha', metavar='A', default=1e-3, type=float, 
    help='Alpha parameter (default: 1e-3).')
parser.add_argument('-q', '--reaction', metavar='Q', default=1.0, type=float, 
    help='Reaction heat coefficient (default: 1.0).')
parser.add_argument('-x', '--xlimits', metavar=('XMIN', 'XMAX'), default=(0, 90), type=float, nargs=2, 
    help='x domain limits (default: [0, 90]).')
parser.add_argument('-y', '--ylimits', metavar=('YMIN', 'YMAX'), default=(0, 90), type=float, nargs=2, 
    help='y domain limits (default: [0, 90]).')
parser.add_argument('-t', '--tlimits', metavar=('TMIN', 'TMAX'), default=(0, 30), type=float, nargs=2, 
    help='t domain limits (default: [0, 30]).')

# Initial conditions 
parser.add_argument('-u0', '--initial-temperature', metavar='U0', type=str, default='',
    help='Initial temperature file. Only .txt and .npy supported (default lambda testing function).')
parser.add_argument('-b0', '--initial-fuel', metavar='B0', type=str, default='',
    help='Initial fuel file. Only .txt and .npy supported (default lambda testing function).')

# Vector field
# Complete in data
parser.add_argument('-vf', '--vector-field', metavar='VF', type=str, default='',
    help='Vector Field. Only .txt and .npy supported (default lambda testing function).')
# Independent files
parser.add_argument('-w', '--wind', metavar='W', type=str, default='',
    help='Wind component. Only .txt and .npy supported (default lambda testing function).')
parser.add_argument('-T', '--terrain', metavar=('Tx', 'Ty'), type=str, default=('', ''), nargs=2,
    help='Topography gradient effect. Only .txt and .npy supported (default no topography effect).')

# Others parameters
parser.add_argument('-acc', '--accuracy', metavar='ACC', default=2, type=int, 
    help='Finite difference accuracy (default: 2).')
parser.add_argument('-sps', '--sparse', metavar='S', default=0, type=int, 
    help='Finite difference sparse matrices (default: 0).')
parser.add_argument('-lst', '--last', metavar='LST', default=1, type=int, 
    help='Only last approximation (default: 1).')
parser.add_argument('-plt', '--plot', metavar='PLT', default=0, type=int,
    help='Plot result (default: 0).')

parser._action_groups.append(optional)
args = parser.parse_args()

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

# Vector field effect
# All info
v_dir = args.vector_field

# Wind
w_dir = args.wind
# Terrain gradient
t_dir = args.terrain

# If u0 is not array
if u0_dir == "":
    u0 = lambda x, y: 6 * G(x-20, y-20, 20)
else:
    print("Loading initial temperature data...")
    u0 = storage.openFile(u0_dir)
    assert u0.shape == (Ny, Nx), "Temprature initial conditions shape and Nx/Ny don't match."

# If b0 is not array
if b0_dir == "":
    b0 = lambda x, y: b0_(x, y)
else:
    print("Loading initial fueld data...")
    b0 = storage.openFile(b0_dir)
    assert b0.shape == (Ny, Nx), "Fuel initial condition size and Nx/Ny don't match."

# If vector field is not data array
if v_dir == "":
    if w_dir != "" and t_dir != ('', ''):
        V = storage.openVectorWT(w_dir, t_dir)
    else:
        # Lambda function example
        gamma = 1
        w1 = lambda x, y, t: gamma * np.cos(np.pi/4) # + 1e-2 * t)
        w2 = lambda x, y, t: gamma * np.sin(np.pi/4) # + 1e-2 * t)
        V = lambda x, y, t: (w1(x, y, t), w2(x, y, t)) #V = (w1, w2)
else:
    print("Loading vector field data...")
    V = storage.openFile(v_dir)
    assert V.shape[0] == (Nt + 1), "File vector size and time steps don't match."

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
print("Starting the numerical simulation...")
t, X, Y, U, B = wildfire_.solvePDE(Nx, Ny, Nt, u0, b0, V, space_method, time_method, last=last, acc=acc, sparse=sparse)
print("Numerical simulation has finished.")

## Plot results
if args.plot:
    print("Plots...")
    # Only last 
    if args.last:
        plots.UB(t, X, Y, U, B, V)
    else: # Some plots
        for n in np.arange(Nt + 1)[::Nt // 4]:
            plots.UBs(n, t, X, Y, U, B, V)