import wildfire
import numpy as np
import matplotlib.pyplot as plt
import plots as p
#from scipy import interpolate

  
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

dx = x[1]-x[0]
    
# Meshes for initial condition plots
X, Y = np.meshgrid(x, y)

# TODO: define parameters in function of real conditions
#T_env = 300
#Ea = 83.68
#A = 1e9
#rho = 1e2
#C = 1
#k = 1

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
alpha = 1e-1#1e-3#1e1#1e-1 # natural convection

# Plot initial conditions
s = 8
#plotScalar(X, Y, u0, "Initial condition", plt.cm.jet)
#plotScalar(X, Y, b0, "Fuel", plt.cm.Oranges)
#plotScalar(X, Y, top, "Topography", plt.cm.Oranges)
#plotField(X[::s,::s], Y[::s,::s], T, "Topography Gradient")
#p.plotField(X[::s,::s], Y[::s,::s], W, "Wind", 0)
#plotField(X[::s,::s], Y[::s,::s], V, "Topography + Wind", 0)
p.plotIC(X, Y, u0, b0, V, W, T, top)

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
    'sparse': True
}

#%%
ct = wildfire.fire(parameters)
#%%
# Finite difference in space
U, B = ct.solvePDE('fd', 'rk4')
#%%
#ct.plots(U, B)

#%% PLOT JCC
p.plotJCC(t, X, Y, U, B, W)

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
s = 10
f, axarr = plt.subplots(4, 2, sharex='col', sharey='row', figsize=(24, 16))
#f.tight_layout()
f.subplots_adjust(left=0.3, right=0.6)

p00 = axarr[0, 0].imshow(U[0], origin='lower', cmap=plt.cm.jet, alpha=0.9, 
     vmin=np.min(U), vmax=np.max(U), extent=[-1, 1, -1, 1])
axarr[0, 0].quiver(X[::s,::s], Y[::s,::s], w1(X[::s,::s], Y[::s,::s], t[0]), w2(X[::s,::s], Y[::s,::s], t[0])) 

p01 = axarr[0, 1].imshow(B[0], origin='lower', cmap=plt.cm.Oranges, alpha=0.9, 
     vmin=np.min(B), vmax=np.max(B), extent=[x[0], x[-1], y[0], y[-1]])

p10 = axarr[1, 0].imshow(U[333], origin='lower', cmap=plt.cm.jet, alpha=0.9, 
     vmin=np.min(U), vmax=np.max(U), extent=[x[0], x[-1], y[0], y[-1]])
axarr[1, 0].quiver(X[::s,::s], Y[::s,::s], w1(X[::s,::s], Y[::s,::s], t[333]), w2(X[::s,::s], Y[::s,::s], t[333])) 


p11 = axarr[1, 1].imshow(B[333], origin='lower', cmap=plt.cm.Oranges, alpha=0.9, 
     vmin=np.min(B), vmax=np.max(B), extent=[x[0], x[-1], y[0], y[-1]])

p20 = axarr[2, 0].imshow(U[666], origin='lower', cmap=plt.cm.jet, alpha=0.9, 
     vmin=np.min(U), vmax=np.max(U), extent=[x[0], x[-1], y[0], y[-1]])
axarr[2, 0].quiver(X[::s,::s], Y[::s,::s], w1(X[::s,::s], Y[::s,::s], t[666]), w2(X[::s,::s], Y[::s,::s], t[666])) 


p21 = axarr[2, 1].imshow(B[666], origin='lower', cmap=plt.cm.Oranges, alpha=0.9, 
     vmin=np.min(B), vmax=np.max(B), extent=[x[0], x[-1], y[0], y[-1]])

p30 = axarr[3, 0].imshow(U[-1], origin='lower', cmap=plt.cm.jet, alpha=0.9, 
     vmin=np.min(U), vmax=np.max(U), extent=[x[0], x[-1], y[0], y[-1]])
axarr[3, 0].quiver(X[::s,::s], Y[::s,::s], w1(X[::s,::s], Y[::s,::s], t[-1]), w2(X[::s,::s], Y[::s,::s], t[-1])) 


p31 = axarr[3, 1].imshow(B[-1], origin='lower', cmap=plt.cm.Oranges, alpha=0.9, 
     vmin=np.min(B), vmax=np.max(B), extent=[x[0], x[-1], y[0], y[-1]])


for ax in axarr.flat:
    #ax.set(xlabel='x', ylabel='y')
    ax.set_xlabel(r"$x$", fontsize=14)
    ax.set_ylabel(r"$y$", fontsize=14)
# Hide x labels and tick labels for top plots and y ticks for right plots.
for ax in axarr.flat:
    ax.label_outer()
    
cb1 = plt.colorbar(p00, ax=axarr[0,0], fraction=0.046, pad=0.04)
cb2 = plt.colorbar(p01, ax=axarr[0,1], fraction=0.046, pad=0.04)
cb3 = plt.colorbar(p10, ax=axarr[1,0], fraction=0.046, pad=0.04)
cb4 = plt.colorbar(p11, ax=axarr[1,1], fraction=0.046, pad=0.04)
cb5 = plt.colorbar(p20, ax=axarr[2,0], fraction=0.046, pad=0.04)
cb6 = plt.colorbar(p21, ax=axarr[2,1], fraction=0.046, pad=0.04)
cb7 = plt.colorbar(p30, ax=axarr[3,0], fraction=0.046, pad=0.04)
cb8 = plt.colorbar(p31, ax=axarr[3,1], fraction=0.046, pad=0.04)

#axarr[3, 1].xticks(fontsize=14)
axarr[0,0].tick_params(axis='both', which='major', labelsize=12)
axarr[1,0].tick_params(axis='both', which='major', labelsize=12)
axarr[2,0].tick_params(axis='both', which='major', labelsize=12)
axarr[3,0].tick_params(axis='both', which='major', labelsize=12)
axarr[3,1].tick_params(axis='both', which='major', labelsize=12)
#plt.tick_params(axis='both', which='minor', labelsize=20)

cb1.ax.tick_params(labelsize=12)
cb2.ax.tick_params(labelsize=12)
cb3.ax.tick_params(labelsize=12)
cb4.ax.tick_params(labelsize=12)
cb5.ax.tick_params(labelsize=12)
cb6.ax.tick_params(labelsize=12)
cb7.ax.tick_params(labelsize=12)
cb8.ax.tick_params(labelsize=12)

plt.rc('axes', labelsize=12)
#plt.savefig('simulation.eps', format='eps', dpi=200, transparent=True, bbox_inches='tight', pad_inches=0)
plt.show()




#%%
# Chebyshev in space
Wc, Bc = ct.solvePDE('cheb', 'rk4')
#%%
ct.plots(Wc, Bc, True)

#%%
W_1024 = np.load('data/last_W_1024.npy')
W_512 = np.load('data/last_W_512.npy')
W_256 = np.load('data/last_W_256.npy')
W_128 = np.load('data/last_W_128.npy')
#%%
#errors = np.array([
#    np.max(np.abs(W_128 - W_1024[::8, ::8])),
#    np.max(np.abs(W_256 - W_1024[::4, ::4])),
#    np.max(np.abs(W_512 - W_1024[::2, ::2]))
#    ])
errors = np.array([
    np.linalg.norm((W_128 - W_1024[::8, ::8]).flatten(), np.inf),
    np.linalg.norm((W_256 - W_1024[::4, ::4]).flatten(), np.inf),
    np.linalg.norm((W_512 - W_1024[::2, ::2]).flatten(), np.inf),
    ])
h = np.array([2/(2**i) for i in range(7, 10)])
#%%
plt.plot(h, errors, '-x')
plt.xlabel("h")
plt.grid(True)
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