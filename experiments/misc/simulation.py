import numpy as np
import sys, datetime, pathlib, time
from wildfire.fire import Fire
#from wildfire import plots as p

# Arguments
L = int(sys.argv[1])
x_ign_n = int(sys.argv[2])
y_ign_n = int(sys.argv[3])
n_exp = sys.argv[4]

SIM_ID = datetime.datetime.now().strftime("%Y%m%d%H%M%S") + n_exp
BASE_DIR = './experiments/misc/test/output/' + SIM_ID + '/'
pathlib.Path(BASE_DIR).mkdir(parents=True, exist_ok=True)

print("Simulation ID: " + SIM_ID)

# Functions
gaussian = lambda A, sigma_x, sigma_y, x, y:  A * np.exp((x**2) / sigma_x + (y**2) / sigma_y)
u0 = lambda x, y: gaussian(6.0, -20.0, -20.0, x, y) 
b0 = lambda x, y: x*0 + 1

# Asensio 2002 experiment
M, N = 128, 128
#L = 500 # Timesteps
x_min, x_max = 0, 90 # x domain limit
y_min, y_max = 0, 90 # y domain limit
t_max = 25

# Ignition domain
x_ign_min, x_ign_max = 20, 70
y_ign_min, y_ign_max = 20, 70

dt = t_max / L

x = np.linspace(x_min, x_max, N) # x domain
y = np.linspace(y_min, y_max, M) # y domain
t = np.linspace(0, t_max, L + 1) # t domain

dx = x[1] - x[0]
dy = y[1] - y[0]

print("Number of numerical simulations: " + str(x_ign_n * y_ign_n))
print("Finite Difference in space")
print('dx: ', dx)
print('dy: ', dy)
print("Euler in time")
print('dt: ', dt)

# Wind effect
gamma = 1
v1 = lambda x, y, t: gamma * np.cos(np.pi/4 + x*0) # 300
v2 = lambda x, y, t: gamma * np.sin(np.pi/4 + x*0) # 300
V = (v1, v2)

# Parameters
kappa = 1e-1 # diffusion coefficient
epsilon = 3e-1#3e-2 # inverse of activation energy
upc = 3 # u phase change
q = 1 # reaction heat
alpha = 1e-3 # natural convection

# Meshes for initial condition plots
X, Y = np.meshgrid(x, y)

# Parameters for the model
parameters = {
    #'u0': lambda x, y: u0(x-20, y-20), 
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
    'complete': False,
    'components': (1, 1, 1)
}


# Ignition points delta
if x_ign_n * y_ign_n > 1:
  dx_ign = (x_ign_max - x_ign_min) / (x_ign_n - 1)
  dy_ign = (y_ign_max - y_ign_min) / (y_ign_n - 1);
else:
  dx_ign = 1
  dy_ign = 1


exe_time = 0.0

for i in range(y_ign_n):
  for j in range(x_ign_n):
    
    # Ignition points
    x_ign = x_ign_min + dx_ign * j;	
    y_ign = y_ign_min + dy_ign * i;
    
    # Parameters
    parameters['u0'] = lambda x, y: u0(x-x_ign, y-y_ign)
    
    # Initial Conditions
    U0 = parameters['u0'](X, Y)
    B0 = b0(X, Y)
    
    # Boundary
    U0[0,:] = np.zeros(N); U0[-1,:] = np.zeros(N)
    U0[:,0] = np.zeros(M); U0[:,-1] = np.zeros(M)
    B0[0,:] = np.zeros(N); B0[-1,:] = np.zeros(N)
    B0[:,0] = np.zeros(M); B0[:,-1] = np.zeros(M)
    
    #np.savetxt(BASE_DIR + "U0_" + str(i) + str(j) + ".txt", U0, delimiter=" ", fmt='%.8f')
    #np.savetxt(BASE_DIR + "B0_" + str(i) + str(j) + ".txt", B0, delimiter=" ", fmt='%.8f')

    # Create wildfire
    ct = Fire(parameters)
    
    start = time.time()
    
    # Solve Finite difference in space, euler in time
    U, B = ct.solvePDE('fd', 'eulveclast')
    
    stop = time.time()
    
    exe_time += (stop - start)
    
    # Save approximation
    #np.savetxt(BASE_DIR + "U_" + str(i) + str(j) + ".txt", U, delimiter=" ", fmt='%.8f')
    #np.savetxt(BASE_DIR + "B_" + str(i) + str(j) + ".txt", B, delimiter=" ", fmt='%.8f')

print("Execution time: ", exe_time, " [s]")

# Write log
log = open(BASE_DIR + 'log.txt', 'w')
log.write("Simulation ID: {0}\n".format(SIM_ID))
log.write("Number of numerical simulations: {0}\n".format(x_ign_n * y_ign_n))
log.write("Parallel approach: None\n")
log.write("\nIgnition points\n")
log.write("----------------\n")
log.write("{0} in x, {1} in y\n".format(x_ign_n, y_ign_n))
log.write("Domain: [{0}, {1}]x[{2}, {3}]\n".format(x_ign_min, x_ign_max, y_ign_min, y_ign_max))
log.write("\nSpace\n")
log.write("------\n")
log.write("Domain: [{0}, {1}]x[{2}, {3}]\n".format(x_min, x_max, y_min, y_max))
log.write("Method: FD\n")
log.write("M: {0}\n".format(M))
log.write("N: {0}\n".format(N))
log.write("dx: {0}\n".format(dx))
log.write("dy: {0}\n".format(dy))
log.write("\nTime\n")
log.write("------\n")
log.write("Domain: [0, {0}]\n".format(t_max))
log.write("Method: Euler\n")
log.write("L: {0}\n".format(L))
log.write("dt: {0}\n".format(dt))
log.write("\nExecution time: {0} [s]\n".format(exe_time))
log.close()