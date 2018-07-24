# Convergence Analysis with Asension 2002 Experiment
import numpy as np
import matplotlib.pyplot as plt
import datetime, pathlib
from wildfire.fire import Fire
from wildfire import plots as p

# Create folder for experiment
now = datetime.datetime.now() 
#SIM_NAME = now.strftime("%Y%m%d%H%M%S")
SIM_NAME = "20180719215925"
DIR_BASE = "/media/dsanmartin/My Passport/Data/Thesis/risk_map/" + SIM_NAME + "/"
#DIR_BASE = "/Volumes/My Passport/Data/Thesis/risk_map/" + SIM_NAME + "/"

#SIM_NAME = "test2"
#DIR_BASE = "/home/dsanmartin/Desktop/" + SIM_NAME + "/"

#pathlib.Path(DIR_BASE).mkdir(parents=True, exist_ok=True)

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
u0 = lambda x, y: 6e0 * G(x+.5, y-.5, 1e-2)

# Fuel initial condition
b0 = lambda x, y: 0.5 * G(x+.75, y-.75, .6) + 0.9 * G(x-.75, y+.75, 1) \
  + 0.016 * G(x+.65, y+.65, .3) + 0.015 * G(x-.65, y-.65, .7) #+ 0.8
# Random
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

X, Y = np.meshgrid(x, y)

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

#ct = wildfire.fire(parameters)

# Times experiment. Generate simulations
Nt = 33
y_t = np.linspace(-.8, .8, Nt)
x_t = y_t[::-1]

#%%
for i in range(Nt):
  for j in range(Nt):
    print("Creating simulation "+ str(i) + ", " + str(j))
    u0 = lambda x, y: 6e0 * G(x+x_t[j], y+y_t[i], 1e-2) # Initial fire
    parameters['u0'] = u0
    np.random.seed(666) # For reproducibility of random fuel
    b0 = lambda x, y: np.round(np.random.uniform(size=(x.shape)), decimals=2) # Initial fuel
    parameters['beta0'] = b0
    ct = Fire(parameters)
    U, B = ct.solvePDE('fd', 'rk4')
  
    #u_name = DIR_BASE + 'U' + str(i) + str(j) + '.npy'
    b_name = DIR_BASE + 'B_' + str(i) + '-' + str(j) + '.npy'
  
    #np.save(u_name, U)
    np.save(b_name, B)
    
#%%
# Get times that % of total fuel is per. 
# Fuel is consumend with fraction fuel is < 0.1.
per = 0.1
f_con = 0.1

times_burnt = np.zeros((Nt, Nt))
burnt_per = np.zeros((Nt, Nt))

for i in range(Nt):
  for j in range(Nt):
    print("Creating times "+ str(i) + ", " + str(j))
    
    B = np.load(DIR_BASE + 'B_' + str(i) + '-' + str(j) + '.npy')
    
    # For burnt fuel plot. (initial - final) / initial
    total_fuel = (np.asarray(B[0,1:-1,1:-1])).sum()
    burnt_fuel = (np.asarray(B[-1,1:-1,1:-1])).sum()
    burnt_per[i, j] = (total_fuel - burnt_fuel) / total_fuel    
    
    # For times plot. Find k where % of fuel is >= 10
    for k in range(len(B)):
      # Count elements < 0.1 without boundary
      burnt = (np.asarray(B[k,1:-1,1:-1]) < f_con).sum()
      if burnt >= int(per * M * N):
        times_burnt[i, j] = t[k]
        break

#%%
# PLOTS 

plt.figure(figsize=(6, 14))
s = 8
X_s, Y_s = X[::s,::s], Y[::s,::s]
X_t, Y_t = np.meshgrid(np.linspace(-1, 1, Nt), np.linspace(-1, 1, Nt)) 
ax1 = plt.subplot(3, 1, 1)

new_cmap = p.truncate_colormap(plt.cm.gray, 0, .05)
fuel = plt.contourf(X, Y, b0(X, Y), cmap=plt.cm.Oranges, alpha=0.5)
topo = plt.contour(X, Y, top(X,Y), vmin=np.min(top(X,Y)), cmap=new_cmap)
plt.clabel(topo, inline=1, fontsize=10)
plt.quiver(X_s, Y_s, W[0](X_s, Y_s, 0), W[1](X_s, Y_s, 0))
cb1 = plt.colorbar(fuel, fraction=0.046, pad=0.04)
plt.ylabel(r"$y$", fontsize=16)
plt.setp(ax1.get_xticklabels(), visible=False)
ax1.tick_params(axis='both', which='major', labelsize=12)

ax2 = plt.subplot(3, 1, 2)

new_hot = p.truncate_colormap(plt.cm.hot, 0.1, .55)
my_cmap = new_hot
my_cmap.set_under('w')
times = plt.imshow(times_burnt, vmin=1e-10, cmap=my_cmap, extent=[-1, 1, -1, 1])
plt.ylabel(r"$y$", fontsize=16)
cb2 = plt.colorbar(times, fraction=0.046, pad=0.04)
plt.setp(ax2.get_xticklabels(), visible=False)
ax2.tick_params(axis='both', which='major', labelsize=12)

ax3 = plt.subplot(3, 1, 3)

#new_hot = p.truncate_colormap(plt.cm.hot, 0, .8)
burn_per = plt.imshow(burnt_per, extent=[-1, 1, -1, 1])
plt.xlabel(r"$x$", fontsize=16)
plt.ylabel(r"$y$", fontsize=16)
cb3 = plt.colorbar(burn_per, fraction=0.046, pad=0.04)
ax3.tick_params(axis='both', which='major', labelsize=12)

plt.tight_layout()
cb1.set_label("Initial Fuel Fraction", size=14)
cb2.set_label("Time to consume 10% of total area", size=14)
cb3.set_label("% of fuel burnt at end of simulation", size=14)

# Save
#plt.savefig(DIR_BASE + SIM_NAME + '.pdf', 
#            format='pdf', dpi=200, transparent=True, bbox_inches='tight', pad_inches=0)

plt.savefig('new_risk_map_simulation_gauss.pdf', 
            format='pdf', dpi=200, transparent=True, bbox_inches='tight', pad_inches=0)

# Show
#plt.show()

