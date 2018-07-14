import wildfire
import numpy as np
import matplotlib.pyplot as plt
import plots as p
#%% First paper experiment
# Domain: [-1, 1]^2 x [0, T*dt]
M, N = 64, 64 # Resolution
L = 1000 # Timesteps
dt = 1e-3 # dt
xa, xb = -1, 1 # x domain limit
ya, yb = -1, 1 # y domain limit
x = np.linspace(xa, xb, N) # x domain
y = np.linspace(ya, yb, M) # y domain
t = np.linspace(0, dt*L, L) # t domain
    
# Meshes for initial condition plots
X, Y = np.meshgrid(x, y)

# Topography test
xi = 1e-1
top = lambda x, y: 1e1*(np.exp(-25*((x+.5)**2 + (y)**2)) + np.exp(-25*((x-.5)**2 + y**2)))
t1 = lambda x, y: xi*-500*((x+.5) * np.exp(-25*((x+.5)**2 + y**2)) + (x-.5) * np.exp(-25*((x-.5)**2 + y**2)))
t2 = lambda x, y: xi*-500*(y * np.exp(-25*((x+.5)**2 + y**2)) + y * np.exp(-25*((x-.5)**2 + y**2)))
#top = lambda x, y: 1e0*(np.exp(-2e-3*((x+30)**2 + (y)**2)) + np.exp(-2e-3*((x-30)**2 + y**2)))
#t1 = lambda x, y: xi*-4e-1**((x+30) * np.exp(-2e-3*((x+30)**2 + y**2)) + (x-30) * np.exp(-2e-3*((x-30)**2 + y**2)))
#t2 = lambda x, y: xi*-4e-1*(y * np.exp(-2e-3*((x+30)**2 + y**2)) + y * np.exp(-2e-3*((x-30)**2 + y**2)))
T = (t1, t2)

# Vector field V = (v1, v2). "Incompressible flow div(V) = 0"
gamma = 1
w1 = lambda x, y, t: gamma * np.cos((7/4+.01*t+x*0)*np.pi)
w2 = lambda x, y, t: gamma * np.sin((7/4+.01*t+y*0)*np.pi)
W = (w1, w2)


# Vector
v1 = lambda x, y, t: w1(x, y, t) + t1(x, y)
v2 = lambda x, y, t: w2(x, y, t) + t2(x, y)
V = (v1, v2)

# Lambda function for temperature initial condition
u0 = lambda x, y: 1e1*np.exp(-150*((x+.7)**2 + (y-.7)**2)) 

# Lambda function for fuel initial condition
b0 = lambda x, y: x*0 + 1 #S(x+.25, y+.25) #x*0 + 1

# Non dimensional parameters
kappa = 1e-2#5e-3 # diffusion coefficient
epsilon = 3e-1#1e-1 # inverse of activation energy
upc = 1#1e-1 # u phase change
q = 1#1e-1 # reaction heat
alpha = 1e-3#1e1#1e-1 # natural convection
#%% TESTING
M, N = 128, 128
L = 1000 # Timesteps
dt = 1e-4 # dt
xa, xb = -1, 1 # x domain limit
ya, yb = -1, 1 # y domain limit
x = np.linspace(xa, xb, N) # x domain
y = np.linspace(ya, yb, M) # y domain
t = np.linspace(0, dt*L, L) # t domain

G = lambda x, y, s: np.exp(-1/s * (x**2 + y**2))
Gx = lambda x, y, s: -2/s * x * np.exp(-1/s * (x**2 + y**2))
Gy = lambda x, y, s: -2/s * y * np.exp(-1/s * (x**2 + y**2))

# Temperature initial condition
u0 = lambda x, y: 6e0 * G(x+.5, y-.5, 1e-2) #9

# Fuel initial condition
b0 = lambda x, y: 0.5 * G(x+.75, y-.75, .6) + 0.9 * G(x-.75, y+.75, 1) \
  + 0.016 * G(x+.65, y+.65, .3) + 0.015 * G(x-.65, y-.65, .7) #+ 0.8

#b0 = lambda x, y: np.round(np.random.uniform(size=(x.shape)), decimals=2)

# Wind effect
gamma = 10
w1 = lambda x, y, t: gamma * np.cos(7/4 * np.pi + x*0)
w2 = lambda x, y, t: gamma * np.sin(7/4 * np.pi + y*0)
W = (w1, w2)

# Topography test
xi = 20
top = lambda x, y: 0.2 * G(x+.5, y+.5, .6) + 0.2 * G(x-.9, y-.9, .9)
t1 = lambda x, y: xi * (0.5 * G(x+.5, y +.6, .6) * Gx(x+.5, y+.5, .6) + 0.5 * G(x-.9, y-.9, .9) * Gx(x-.9, y-.9, .9))
t2 = lambda x, y: xi * (0.5 * G(x+.5, y+.5, .6) * Gy(x+.5, y+.5, .6) + 0.5 * G(x-.9, y-.9, .9) * Gy(x-.9, y-.9, .9))
T = (t1, t2)

# Vector
v1 = lambda x, y, t: w1(x, y, t) + t1(x, y)
v2 = lambda x, y, t: w2(x, y, t) + t2(x, y)
V = (v1, v2)

# Parameters
kappa = 1e-2 # diffusion coefficient
epsilon = 1e-1#3e-2 # inverse of activation energy
upc = 1e0 # u phase change
q = 5e-3 # reaction heat
alpha = 1e-3#1e-3 # natural convection
#%% Asensio 2002 experiment
M, N = 128, 128
L = 500#3000 # Timesteps
dt = 1e-2 # dt
xa, xb = 0, 90 # x domain limit
ya, yb = 0, 90 # y domain limit
x = np.linspace(xa, xb, N) # x domain
y = np.linspace(ya, yb, M) # y domain
t = np.linspace(0, dt*L, L) # t domain

# Temperature initial condition
u0 = lambda x, y: 6e0*np.exp(-5e-2*((x-20)**2 + (y-20)**2)) 

# Fuel initial condition
np.random.seed(666)
b0 = lambda x, y: np.round(np.random.uniform(size=(x.shape)), decimals=2)#x*0 + 1 #S(x+.25, y+.25) #x*0 + 1

# Wind effect
gamma = 1
w1 = lambda x, y, t: gamma * np.cos(np.pi/4 + x*0) # 300
w2 = lambda x, y, t: gamma * np.sin(np.pi/4 + x*0) # 300
W = (w1, w2)

# Topography test
xi = 0#1e-1
top = lambda x, y: 1e0*(np.exp(-2e-3*((x+30)**2 + (y)**2)) + np.exp(-2e-3*((x-30)**2 + y**2)))
t1 = lambda x, y: xi*-4e-1**((x+30) * np.exp(-2e-3*((x+30)**2 + y**2)) + (x-30) * np.exp(-2e-3*((x-30)**2 + y**2)))
t2 = lambda x, y: xi*-4e-1*(y * np.exp(-2e-3*((x+30)**2 + y**2)) + y * np.exp(-2e-3*((x-30)**2 + y**2)))
T = (t1, t2)

# Vector
v1 = lambda x, y, t: w1(x, y, t) + t1(x, y)
v2 = lambda x, y, t: w2(x, y, t) + t2(x, y)
V = (v1, v2)

# Parameters
kappa = 1e-1 # diffusion coefficient
epsilon = 3e-1#3e-2 # inverse of activation energy
upc = 3e0 # u phase change
q = 1 # reaction heat
alpha = 1e-3 # natural convection
#%% Mell 2006 experiment
M, N = 128, 128
L = 3000 # Timesteps
dt = 1e-2 # dt
xa, xb = -100, 100 # x domain limit
ya, yb = -100, 100 # y domain limit
x = np.linspace(xa, xb, N) # x domain
y = np.linspace(ya, yb, M) # y domain
t = np.linspace(0, dt*L, L) # t domain

# Temperature initial condition
u0 = lambda x, y: 6e0*np.exp(-2e-2*((x+90)**2 + (y)**2)) #1e2
#u0 = lambda x, y: 6e1*(np.zeros((x.shape)) + np.ones((x.shape[0], 2)))
#def u0(x, y):
#  out = np.zeros((x.shape))
#  #out[20:-20, :5] = 6*np.ones((x.shape[0]-40, 5))
#  out[35:-35, :4] = 6*np.ones((x.shape[0]-70, 4))
#  #out[5:15, :4] = 6*np.ones((10, 4))
#  #out[-15:-5, :4] = 6*np.ones((10, 4))
#  return out

# Fuel initial condition
#b0 = lambda x, y: x*0 + 1
b0 = lambda x, y: np.round(np.random.uniform(size=(x.shape)), decimals=2)

# Wind effect
gamma = 2.5 #1
w1 = lambda x, y, t: gamma * np.cos(0 + x*0)
w2 = lambda x, y, t: gamma * np.sin(0 + x*0)
W = (w1, w2)

# Topography test
xi = 0#1e-1
top = lambda x, y: 1e0*(np.exp(-2e-3*((x+30)**2 + (y)**2)) + np.exp(-2e-3*((x-30)**2 + y**2)))
t1 = lambda x, y: xi*-4e-1**((x+30) * np.exp(-2e-3*((x+30)**2 + y**2)) + (x-30) * np.exp(-2e-3*((x-30)**2 + y**2)))
t2 = lambda x, y: xi*-4e-1*(y * np.exp(-2e-3*((x+30)**2 + y**2)) + y * np.exp(-2e-3*((x-30)**2 + y**2)))
T = (t1, t2)

# Vector
v1 = lambda x, y, t: w1(x, y, t) + t1(x, y)
v2 = lambda x, y, t: w2(x, y, t) + t2(x, y)
V = (v1, v2)

# Parameters
kappa = 1e1 # diffusion coefficient
epsilon = 3e-1#3e-2 # inverse of activation energy
upc = 1e0 # u phase change
q = 3#1 # reaction heat
alpha = 1e-2#1e-4 # natural convection
#%%
# Meshes for initial condition plots
#X, Y = np.meshgrid(x, y)

# Plot initial conditions
#p.plotIC(X, Y, u0, b0, V, W, T=None, top=None)

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
    'sparse': True,
    'show': False,
    'complete': False
}

ct = wildfire.fire(parameters)
#%%
# Finite difference in space
U, B = ct.solvePDE('fd', 'rk4')
#timeit U, B = ct.solvePDE('fd', 'last')
#%%
ct.plots(U, B)

#%% PLOT JCC
p.plotJCC(t, X, Y, U, B, W, T=T, save=False)

#%% BURNT RATE PLOT
plt.figure(figsize=(6, 4))

dif_b = (B[1:] - B[:-1]) / dt
levels = np.arange(-0.04, np.max(dif_b), 0.01)

row = 5
tim = L // (row - 1)

for i in range(row):
  tt = i*tim
  if i == (row - 1):
    tt = -1
  plt.contourf(X, Y, dif_b[tt], levels=levels)

plt.xlabel(r"$x$", fontsize=16)
plt.ylabel(r"$y$", fontsize=16)
plt.colorbar()
#plt.show()
plt.savefig('burnt_rate.eps', format='eps', dpi=200, transparent=True, bbox_inches='tight', pad_inches=0)


#%% Times experiment
x_t = np.array([0.8, 0.4, 0.0, -0.4, -0.8])
y_t = np.array([-0.8, -0.4, 0.0, 0.4, 0.8])

for i in range(5):
  for j in range(5):
    u0 = lambda x, y: 1e1*np.exp(-150*((x+x_t[j])**2 + (y+y_t[i])**2)) 
    parameters['u0'] = u0
    ct = wildfire.fire(parameters)
    U, B = ct.solvePDE('fd', 'rk4')
    
    u_name = 'experiments/times/U' + str(i) + str(j) + '.npy'
    b_name = 'experiments/times/B' + str(i) + str(j) + '.npy'
    
    np.save(u_name, U)
    np.save(b_name, B)
    
#%%
    
per = 0.1
times_burnt = np.zeros((5, 5))

for i in range(5):
  for j in range(5):
    
    B = np.load('experiments/times/B' + str(i) + str(j) + '.npy')
    
    for k in range(len(B)):
      # Count elements < 0.5 without boundary
      burnt = (np.asarray(B[k,1:-2,1:-2]) < 0.5).sum()
      
      if burnt >= int(per * 64**2):
        times_burnt[i, j] = t[k]
        break

#plt.imshow(B[-1])
plt.imshow(times_burnt, extent=[-1, 1, -1, 1])
plt.colorbar()
#%%
# Chebyshev in space
Wc, Bc = ct.solvePDE('cheb', 'rk4')
#%%
ct.plots(Wc, Bc, True)

#%%
U_1024 = np.load('experiments/convergence/500/U_1024.npy')
U_512 = np.load('experiments/convergence/500/U_512.npy')
U_256 = np.load('experiments/convergence/500/U_256.npy')
U_128 = np.load('experiments/convergence/500/U_128.npy')
U_64 = np.load('experiments/convergence/500/U_64.npy')

B_1024 = np.load('experiments/convergence/500/B_1024.npy')
B_512 = np.load('experiments/convergence/500/B_512.npy')
B_256 = np.load('experiments/convergence/500/B_256.npy')
B_128 = np.load('experiments/convergence/500/B_128.npy')
B_64 = np.load('experiments/convergence/500/B_64.npy')
#%%
errors_u = np.array([
    np.linalg.norm((U_64 - U_1024[::16, ::16]).flatten(), np.inf),
    np.linalg.norm((U_128 - U_1024[::8, ::8]).flatten(), np.inf),
    np.linalg.norm((U_256 - U_1024[::4, ::4]).flatten(), np.inf),
    np.linalg.norm((U_512 - U_1024[::2, ::2]).flatten(), np.inf),
    ])

errors_b = np.array([
    np.linalg.norm((B_64 - B_1024[::16, ::16]).flatten(), np.inf),
    np.linalg.norm((B_128 - B_1024[::8, ::8]).flatten(), np.inf),
    np.linalg.norm((B_256 - B_1024[::4, ::4]).flatten(), np.inf),
    np.linalg.norm((B_512 - B_1024[::2, ::2]).flatten(), np.inf),
    ])
  
h = np.array([90/(2**i) for i in range(6, 10)])
#%%
#plt.plot(h, errors_u, 'b-x')
plt.plot(h, errors_b, 'r-o')
plt.xlabel("h")
plt.grid(True)
plt.xscale('log')
plt.yscale('log')
plt.show()

#%%
MN = np.array([1024, 4096, 16384, 65536])
mean_time = np.array([0.467, 0.978, 3.55, 18.2])
std_time = np.array([0.282, 0.0718, 0.136, 0.248])
plt.plot(MN, mean_time, 's-r')
plt.plot(MN, mean_time + std_time)
plt.plot(MN, mean_time - std_time)
plt.grid(True)
plt.xscale('log')
plt.yscale('log')
plt.show()
#%%
plt.contour(X,Y,U[0])
plt.contour(X,Y,U[3000])
plt.contour(X,Y,U[4000])
plt.contour(X,Y,U[5000-1])
plt.show()