import wildfire
import numpy as np
import matplotlib.pyplot as plt
import plots as p
#%%
# Domain: [-1, 1]^2 x [0, T*dt]
#M, N = 16, 16 # Resolution
M, N = 128, 128
L = 5000 # Timesteps
dt = 1e-3 # dt
xa, xb = -100, 100 # x domain limit
ya, yb = -100, 100 # y domain limit
x = np.linspace(xa, xb, N) # x domain
y = np.linspace(ya, yb, M) # y domain
t = np.linspace(0, dt*L, L) # t domain
    
# Meshes for initial condition plots
X, Y = np.meshgrid(x, y)

# Topography test
xi = 0#1e-1
#top = lambda x, y: 1e1*(np.exp(-25*((x+.5)**2 + (y)**2)) + np.exp(-25*((x-.5)**2 + y**2)))
#t1 = lambda x, y: xi*-500*((x+.5) * np.exp(-25*((x+.5)**2 + y**2)) + (x-.5) * np.exp(-25*((x-.5)**2 + y**2)))
#t2 = lambda x, y: xi*-500*(y * np.exp(-25*((x+.5)**2 + y**2)) + y * np.exp(-25*((x-.5)**2 + y**2)))
top = lambda x, y: 1e0*(np.exp(-2e-3*((x+30)**2 + (y)**2)) + np.exp(-2e-3*((x-30)**2 + y**2)))
t1 = lambda x, y: xi*-4e-1**((x+30) * np.exp(-2e-3*((x+30)**2 + y**2)) + (x-30) * np.exp(-2e-3*((x-30)**2 + y**2)))
t2 = lambda x, y: xi*-4e-1*(y * np.exp(-2e-3*((x+30)**2 + y**2)) + y * np.exp(-2e-3*((x-30)**2 + y**2)))
T = (t1, t2)

# Vector field V = (v1, v2). "Incompressible flow div(V) = 0"
gamma = 1
#w1 = lambda x, y, t: gamma * np.cos((7/4+.01*t)*np.pi)
#w2 = lambda x, y, t: gamma * np.sin((7/4+.01*t)*np.pi)
#w1 = lambda x, y, t: gamma * np.cos(-np.pi/2)
#w2 = lambda x, y, t: gamma * np.sin(-np.pi/2)
w1 = lambda x, y, t: gamma * np.cos(0)
w2 = lambda x, y, t: gamma * np.sin(0)
#w1 = lambda x, y, t: gamma * np.cos(np.pi/4)
#w2 = lambda x, y, t: gamma * np.sin(np.pi/4)
W = (w1, w2)


# Vector
v1 = lambda x, y, t: w1(x, y, t) + t1(x, y)
v2 = lambda x, y, t: w2(x, y, t) + t2(x, y)
V = (v1, v2)

# Lambda function for temperature initial condition
#u0 = lambda x, y: 1e1*np.exp(-150*((x+.8)**2 + (y-.8)**2)) 
#u0 = lambda x,y: 1e1*np.exp(-(5e3*(x+0.9)**2 + 1e1*(y)**2))
#u0 = lambda x, y: 6e1*np.exp(-9e-3*((x+60)**2 + (y+60)**2)) 
#u0 = lambda x, y: 6e1*np.exp(-1e2*((x+.7)**2 + (y+.7)**2)) 
u0 = lambda x,y: 6e1*np.exp(-(1e-1*(x+90)**2 + 1e-3*(y)**2))

# Lambda function for fuel initial condition
b0 = lambda x, y: x*0 + 1 #S(x+.25, y+.25) #x*0 + 1

# Non dimensional parameters
#kappa = 1e-2#5e-3 # diffusion coefficient
#epsilon = 1e-1 # inverse of activation energy
#upc = 1e-1 # u phase change
#q = 1e-1 # reaction heat
#alpha = 1e1#1e1#1e-1 # natural convection

kappa = 1e-1#5e-3 # diffusion coefficient
epsilon = 3e-1#3e-1 # inverse of activation energy
upc = 3e1 # u phase change
q = 1e-0 # reaction heat
alpha = 1e-1 # natural convection
#%% Asensio 2002 experiment
#M, N = 128, 128
M, N = 256, 256
L = 2000 # Timesteps
dt = 1e-3 # dt
xa, xb = -100, 100 # x domain limit
ya, yb = -100, 100 # y domain limit
x = np.linspace(xa, xb, N) # x domain
y = np.linspace(ya, yb, M) # y domain
t = np.linspace(0, dt*L, L) # t domain

# Temperature initial condition
u0 = lambda x, y: 6e1*np.exp(-2e-2*((x+60)**2 + (y+60)**2)) 

# Fuel initial condition
b0 = lambda x, y: np.round(np.random.uniform(size=(x.shape)), decimals=2)#x*0 + 1 #S(x+.25, y+.25) #x*0 + 1

# Wind effect
gamma = 1
w1 = lambda x, y, t: gamma * np.cos(np.pi/4 + x*0)
w2 = lambda x, y, t: gamma * np.sin(np.pi/4 + x*0)
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
M, N = 64, 64
L = 1000 # Timesteps
dt = 1e-2 # dt
xa, xb = -100, 100 # x domain limit
ya, yb = -100, 100 # y domain limit
x = np.linspace(xa, xb, N) # x domain
y = np.linspace(ya, yb, M) # y domain
t = np.linspace(0, dt*L, L) # t domain

# Temperature initial condition
u0 = lambda x, y: 1e2*np.exp(-2e-2*((x+80)**2 + (y)**2)) 

# Fuel initial condition
b0 = lambda x, y: x*0 + 1

# Wind effect
gamma = 1
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
q = 1e1 # reaction heat
alpha = 1e-2 # natural convection
#%%
# Meshes for initial condition plots
X, Y = np.meshgrid(x, y)

# Plot initial conditions
p.plotIC(X, Y, u0, b0, V, W, T, top)

# Parameters for the model
parameters = {
    'u0': u0, 
    'beta0': b0,
    'v': W,
    'kappa': kappa, 
    'epsilon': epsilon, 
    'upc': upc, 
    'q': q, 
    'alpha': alpha, 
    'x': x, 
    'y': y,
    't': t,
    'sparse': True
}

ct = wildfire.fire(parameters)
#%%
# Finite difference in space
U, B = ct.solvePDE('fd', 'rk4')
#ct.plots(U, B)

#%% PLOT JCC
p.plotJCC(t, X, Y, U, B, W, save=False)

#%%
plt.figure(figsize=(10, 10))
#plt.contourf(X, Y, U[0])
#plt.contourf(X, Y, U[250])
#plt.contourf(X, Y, U[500])
#plt.contourf(X, Y, U[750])
#plt.contourf(X, Y, U[-1])
#levels = np.arange(3, 5, 0.1)
#plt.contour(X, Y, B[0])
#plt.contour(X, Y, B[250], levels=levels)
#plt.contour(X, Y, B[500], levels=levels)
#plt.contour(X, Y, B[750], levels=levels)
#plt.contour(X, Y, B[-1], levels=levels)
#Bs = (1-B[0]) + (1-B[250]) + (1-B[500]) + (1-B[750]) + (1-B[-1])
#plt.contourf(X, Y, Bs)#, levels=levels)

dif_b = (B[1:] - B[:-1]) / dt
levels = np.arange(-0.07, -0.04, 0.01)
print(levels)
#plt.contourf(dif_b[500])
plt.contourf(X, Y, dif_b[0], levels=levels)
plt.contourf(X, Y, dif_b[250], levels=levels)
plt.contourf(X, Y, dif_b[500], levels=levels)
plt.contourf(X, Y, dif_b[750], levels=levels)
plt.contourf(X, Y, dif_b[-1], levels=levels)


#levels = np.arange(0.5, 1.0, 0.1)
#plt.imshow(B[250], origin="lower", extent=[-100, 100, -100, 100])
#plt.imshow(B[500])
#plt.imshow(B[750])
#plt.imshow(B[-1])
#plt.contour(B[250], levels, origin="lower", extent=[-100, 100, -100, 100])
plt.colorbar()
plt.show()

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
U_1024 = np.load('experiments/convergence/last_u_1024.npy')
U_512 = np.load('experiments/convergence/last_u_512.npy')
U_256 = np.load('experiments/convergence/last_u_256.npy')
U_128 = np.load('experiments/convergence/last_u_128.npy')
U_64 = np.load('experiments/convergence/last_u_64.npy')

B_1024 = np.load('experiments/convergence/last_b_1024.npy')
B_512 = np.load('experiments/convergence/last_b_512.npy')
B_256 = np.load('experiments/convergence/last_b_256.npy')
B_128 = np.load('experiments/convergence/last_b_128.npy')
B_64 = np.load('experiments/convergence/last_b_64.npy')
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
  
h = np.array([200/(2**i) for i in range(6, 10)])
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
mean_time = np.array([1.59, 3.37, 9.58, 43.8])
std_time = np.array([0.0565, 0.00761, 0.0261, 1.6])
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