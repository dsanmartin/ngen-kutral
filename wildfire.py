import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp2d
import pathlib, json, inspect, os
from datetime import datetime

sec = int(datetime.today().timestamp())
DIR_BASE = "simulation/" + str(sec) + "/"

# Chebyshev differentiation matrix
def chebyshevMatrix(N):
    if N == 0:
        D = 0
        x = 1
        return D, x
    x = np.cos(np.pi * np.arange(N + 1) / N)
    c = np.hstack((2, np.ones(N - 1), 2)) * ((-1.)**np.arange(N + 1))
    X = np.tile(x, (N + 1, 1)).T
    dX = X - X.T
    D = np.outer(c, 1./c) / (dX + np.eye(N + 1))
    D = D - np.diag(np.sum(D.T, axis=0))
    return D, x

class fire:
  
  def __init__(self, parameters):
    self.u0 = parameters['u0']
    self.beta0 = parameters['beta0']
    self.kappa = parameters['kappa']
    self.epsilon = parameters['epsilon']
    self.upc = parameters['upc']
    self.q = parameters['q']
    self.v = parameters['v']
    self.alpha = parameters['alpha'] 
    self.x = parameters['x']
    self.y = parameters['y']
    self.t = parameters['t']
    self.N, self.M = len(self.x), len(self.y)
    self.T = len(self.t)
    self.dx = self.x[1] - self.x[0]
    self.dy = self.y[1] - self.y[0]
    self.dt = self.t[1] - self.t[0]
    

  # Compute divergence 
  def div(self, F):
    """
    Divergence of F=(f1, f2). div(F) = f1_x + f2_y
    """
    # Get vector field elements
    f1, f2 = F
    
    # Computing df1/dx and df2/dy
    f1_x = np.gradient(f1, self.dx, edge_order=2, axis=1)
    f2_y = np.gradient(f2, self.dy, edge_order=2, axis=0)
    
    return f1_x + f2_y # Divergence (d/dx, d/dy) dot (f1, f2)
  
  # Compute Gradient of f (fx, fy)
  def grad(self, f):
    """
    Gradient of f. grad(f) = (f_x, f_y)
    """
    # Computing df/dx and df/dy
    f_x = np.gradient(f, self.dx, edge_order=2, axis=1)
    f_y = np.gradient(f, self.dy, edge_order=2, axis=0)
    
    return (f_x, f_y) # Gradient (df/dx, df/dy)
    
    
  # Compute Laplacian 
  def laplacian(self, u):
    """
    Laplacian of u. div(grad u)
    """
    return self.div(self.grad(u)) 
            
  # RHS of PDE
  def RHS(self, U, B, V, args=None):
    """
    Compute right hand side of PDE
    """
    
    V1, V2 = V # Unpacking tuple of Vector field
    
    if args is None: # Use finite difference
      Ux, Uy = self.grad(U) # Compute gradient
      # divV = self.div(V) # Compute divergence of V. 
      # This should be 0 for an incompressible flow. This is an assumption for the model.      
      diffusion = (self.kappa * self.laplacian(U)) # k nabla u
      #convection = U * divV + ux*v1 + uy*v2 # div(uV) = u div(F) + V dot grad u
      convection = Ux*V1 + Uy*V2     
      fuel = self.f(U, B)
      
    else: # Use Chebyshev
      Dx, Dy, D2x, D2y = args
      diffusion = self.kappa*(np.dot(U, D2x.T) + np.dot(D2y, U))
      convection = np.dot(U, Dx.T) * V1 + U * np.dot(V1, Dx.T) \
          + np.dot(Dy, U) * V2 + U * np.dot(Dy, V2)            
      fuel = self.f(U, B)
    
    return diffusion - convection + fuel
  
  def K(self, u):
    return self.kappa * (1 + self.epsilon * u) ** 3 + 1
  
  def f(self, u, beta):
    return self.s(u) * beta * np.exp(u / (1 + self.epsilon*u)) - self.alpha * u
  
  def g(self, u, beta):
    return -self.s(u) * (self.epsilon / self.q) * beta * np.exp(u /(1 + self.epsilon*u))
    
  def s(self, u):
    S = np.zeros_like(u)
    S[u >= self.upc] = 1
    
    return S
  
  # Runge-Kutta 4th order for time 
  def solveRK4(self, U0, B0, V, args):
    M, N = U0.shape
    
    U = np.zeros((self.T+1, M, N))
    B = np.zeros((self.T+1, M, N))
    
    U[0] = U0
    B[0] = B0
    
    for t in range(1, self.T + 1):
      k1 = self.RHS(U[t-1], B[t-1], V, args)
      k2 = self.RHS(U[t-1] + 0.5*self.dt*k1, B[t-1] + 0.5*self.dt*k1, V, args)
      k3 = self.RHS(U[t-1] + 0.5*self.dt*k2, B[t-1] + 0.5*self.dt*k2, V, args)
      k4 = self.RHS(U[t-1] + self.dt*k3, B[t-1] + self.dt*k3, V, args)

      U[t] = U[t-1] + (1/6)*self.dt*(k1 + 2*k2 + 2*k3 + k4)
      
      # BC of temperature
      U[t,0,:] = np.zeros(N)
      U[t,-1,:] = np.zeros(N)
      U[t,:,0] = np.zeros(M)
      U[t,:,-1] = np.zeros(M)
      
      bk1 = self.g(U[t-1], B[t-1])
      bk2 = self.g(U[t-1] + 0.5*self.dt*bk1, B[t-1] + 0.5*self.dt*bk1)
      bk3 = self.g(U[t-1] + 0.5*self.dt*bk2, B[t-1] + 0.5*self.dt*bk2)
      bk4 = self.g(U[t-1] + self.dt*bk3, B[t-1] + self.dt*bk3)

      B[t] = B[t-1] + (1/6)*self.dt*(bk1 + 2*bk2 + 2*bk3 + bk4)
      
      # BC of fuel
      B[t,0,:] = np.zeros(N)
      B[t,-1,:] = np.zeros(N)
      B[t,:,0] = np.zeros(M)
      B[t,:,-1] = np.zeros(M)
      
    return U, B
      
  # Forward Euler for time
  def solveEuler(self, U0, B0, V, args):
    M, N = U0.shape
    
    U = np.zeros((self.T+1, M, N))
    B = np.zeros((self.T+1, M, N))
    
    U[0] = U0
    B[0] = B0
    
    for t in range(1, self.T + 1):
      U[t] = U[t-1] + self.RHS(U[t-1], B[t-1], V, args) * self.dt
      B[t] = B[t-1] + self.g(U[t-1], B[t-1]) * self.dt
      
      U[t,0,:] = np.zeros(N)
      U[t,-1,:] = np.zeros(N)
      U[t,:,0] = np.zeros(M)
      U[t,:,-1] = np.zeros(M)
      
      B[t,0,:] = np.zeros(N)
      B[t,-1,:] = np.zeros(N)
      B[t,:,0] = np.zeros(M)
      B[t,:,-1] = np.zeros(M)
      
    return U, B
          
  
  # Solve PDE
  def solvePDE(self, spatial='fd', time='rk4'):
    """
    Solve PDE model
    
    Parameters
    ----------
    spatial : string
            * 'fd' Finite difference
            * 'cheb' Chebyshev differenciation  
    time : string
          * 'euler' Forward Euler
          * 'rk4' Runge-Kutta 4th order
            
    Returns
    -------
    U : (T, M, N) ndarray
      Temperatures approximation
    B : (T, M, N) ndarray  
      Fuels approximation
    """
        
    if spatial == 'cheb':
      Dx, x = chebyshevMatrix(self.N-1)
      Dy, y = chebyshevMatrix(self.M-1)
      
      D2x = np.dot(Dx, Dx)
      D2y = np.dot(Dy, Dy)
      
      X, Y = np.meshgrid(x, y)      
      
      U0 = self.u0(X, Y)
      B0 = self.beta0(X, Y)
      
      V1 = self.v[0](X, Y)
      V2 = self.v[1](X, Y)
      
      V = (V1, V2)
      
      args = (Dx, Dy, D2x, D2y)
      
      
    elif spatial == "fd":
      # Grid for functions evaluation
      X, Y = np.meshgrid(self.x, self.y)
      
      U0 = self.u0(X, Y) # Temperature initial condition
      B0 = self.beta0(X, Y) # Fuel initial condition
      V = (self.v[0](X, Y), self.v[1](X, Y)) # Vector field
      
      args = None
    else:
      print("Spatial method error")
    
    # Time
    if time == 'rk4':
      U, B = self.solveRK4(U0, B0, V, args)
    elif time == 'euler': 
      U, B = self.solveEuler(U0, B0, V, args)
    else:
      print("Time method error")
        
    return U, B
  

  def solveSPDE1(self, sigma):
    # Solve
    U = np.zeros((self.T+1, self.M, self.N))
    U[0] = self.u0
    
    for i in range(1, self.T+1):
        W =  self.F(U[i-1], self.t, self.mu)
        U[i] = U[i-1] + W*self.dt + sigma*np.random.normal(0, self.dt, W.shape)
    
    return U
    

  def solveSPDE2(self, sigma):
    # Solve
    U = np.zeros((self.T+1, self.M, self.N))
    U[0] = self.u0
    for i in range(1, self.T+1):
        W =  self.F(U[i-1], self.t, self.mu)
        U[i] = U[i-1] + W*self.dt + self.dt*sigma*np.random.normal(0, 1, W.shape)*W/self.mu
    
    return U
    

  def plotTemperatures(self, t, temperatures):
    fine = np.linspace(self.x[-1], self.x[1], 2*self.N)
    fu = interp2d(self.x, self.y, temperatures[t], kind='cubic')
    U = fu(fine, fine)
    #U = temperatures[t].reshape(self.u0.shape)
    plt.imshow(U, origin='lower', cmap=plt.cm.jet, extent=[self.x[0], self.x[-1],
               self.y[0], self.y[-1]])
    plt.colorbar()
    #Xf, Yf = np.meshgrid(fine, fine)
    #cont = plt.contourf(Xf, Yf, U, cmap=plt.cm.jet, alpha=0.4)
    #plt.colorbar(cont)
    fig_n = t//10 + 1
    if fig_n < 10:
      fig_name = '0' + str(fig_n)
    else:
      fig_name = str(fig_n)
      
    plt.savefig('simulation/' + fig_name + '.png')
    plt.show()
    
  def plotTemperaturesCheb(self, t, temperatures):
    N = temperatures[t].shape[0]
    fine = np.linspace(-1, 1, 2*N)
    _, x = chebyshevMatrix(N-1)
    _, y = chebyshevMatrix(N-1)
    fu = interp2d(x, y, temperatures[t], kind='cubic')
    U = fu(fine, fine)
    plt.imshow(U, origin='lower', cmap=plt.cm.jet, extent=[-1, 1, -1, 1])
    plt.colorbar()
    plt.show()
    
  def plotFuel(self, t, fuel, save=False):
    X, Y = np.meshgrid(self.x, self.y)
    #fine = np.linspace(self.x[0], self.x[-1], 2*self.N)
    #fu = interp2d(self.x, self.y, fuel[t], kind='cubic')
    #U = fu(fine, fine)
    B = fuel[t]
    plt.imshow(B, origin='lower', cmap=plt.cm.Oranges, alpha=1, 
               extent=[self.x[0], self.x[-1], self.y[0], self.y[-1]])
    plt.colorbar()  

    if save:
      fig_n = t//10 + 1
      if fig_n < 10:
        fig_name = '0' + str(fig_n)
      else:
        fig_name = str(fig_n)       
      pathlib.Path(DIR_BASE + "figures/fuel").mkdir(parents=True, exist_ok=True) 
      plt.savefig(DIR_BASE + "figures/fuel/" + fig_name + '.png')
      
    plt.show()
    
  def plotSimulation(self, t, temperatures, save=False):
    X, Y = np.meshgrid(self.x, self.y)
    #fine = np.linspace(self.x[0], self.x[-1], 2*self.N)
    #fu = interp2d(self.x, self.y, temperatures[t], kind='cubic')
    #U = fu(fine, fine)
    U = temperatures[t]
    plt.imshow(U, origin='lower', cmap=plt.cm.jet, alpha=0.9, 
               extent=[self.x[0], self.x[-1], self.y[0], self.y[-1]])
    plt.colorbar()
    
    Xv, Yv = np.mgrid[self.x[0]:self.x[-1]:complex(0, self.N // 2), 
                      self.y[0]:self.y[-1]:complex(0, self.M // 2)]
    plt.quiver(Xv, Yv, self.v[0](Xv, Yv), self.v[1](Xv, Yv))      

    if save:
      fig_n = t//10 + 1
      if fig_n < 10:
        fig_name = '0' + str(fig_n)
      else:
        fig_name = str(fig_n)        
      pathlib.Path(DIR_BASE + "figures/temperature").mkdir(parents=True, exist_ok=True) 
      plt.savefig(DIR_BASE + "figures/temperature/" + fig_name + '.png')
      
    plt.show()
    
  def plots(self, U, B, cheb=False, save=False):
    
    sec = int(datetime.today().timestamp())
    DIR_BASE = "simulation/" + str(sec) + "/"
    
    fineX = np.linspace(self.x[0], self.x[-1], 2*self.N)    
    fineY = np.linspace(self.y[0], self.y[-1], 2*self.M) 
    
    
    for i in range(self.T):
      if i % 10 == 0:
        if i == 0: kind_ = "linear"
        else: kind_ = "cubic"
        
        if cheb:
          _, x = chebyshevMatrix(self.N-1)
          _, y = chebyshevMatrix(self.M-1)
          fu = interp2d(x, y, U[i], kind='cubic')          
          fb = interp2d(x, y, B[i], kind=kind_)
        else:
          fu = interp2d(self.x, self.y, U[i], kind='cubic')
          fb = interp2d(self.x, self.y, B[i], kind=kind_)
          
        Ui = fu(fineX, fineY)
        Bi = fb(fineX, fineY)

        # Left plot        
        plt.subplot(1, 2, 1) 
        
        #fig = plt.figure(figsize=(14, 8)) 
        
        Xf, Yf = np.meshgrid(fineX, fineY)
        # Temperature plot
        temp = plt.imshow(Ui, origin='lower', cmap=plt.cm.jet, alpha=0.8, vmin=np.min(U),
                   vmax=np.max(U), extent=[self.x[0], self.x[-1], self.y[0], self.y[-1]])
        #temp = plt.contour(Xf, Yf, Ui, cmap=plt.cm.jet, alpha=0.8)
        #fig.colorbar(temp)#, 
        plt.colorbar(temp, fraction=0.046, pad=0.04)
        
        
        #fuel_cont = plt.contour(Xf, Yf, Bi, cmap=plt.cm.Oranges, alpha=0.8)
        #fig.colorbar(fuel_cont)#, fraction=0.046, pad=0.04)
        
        # Wind plot
        Xv, Yv = np.mgrid[self.x[0]:self.x[-1]:complex(0, self.N // 2), 
                          self.y[0]:self.y[-1]:complex(0, self.M // 2)]
        plt.quiver(Xv, Yv, self.v[0](Xv, Yv), self.v[1](Xv, Yv)) 
        
        plt.title("Temperature + Wind")
        plt.xlabel("x")
        plt.ylabel("y")

        # Right plot        
        plt.subplot(1, 2, 2)
        
        # Fuel plot
        plt.imshow(Bi, origin='lower', cmap=plt.cm.Oranges, alpha=1, vmin=np.min(B),
                   vmax=np.max(B), extent=[self.x[0], self.x[-1], self.y[0], self.y[-1]])
        plt.colorbar(fraction=0.046, pad=0.04)  
        plt.title("Fuel")
        plt.xlabel("x")
        plt.ylabel("y")
        
        plt.tight_layout()
        
        if save:
          fig_n = i // 10 + 1
          if fig_n < 10:
            fig_name = '00' + str(fig_n)
          elif fig_n < 100:
            fig_name = '0' + str(fig_n)
          else:
            fig_name = str(fig_n)             
            
          pathlib.Path(DIR_BASE + "figures/sims/").mkdir(parents=True, exist_ok=True) 
          plt.savefig(DIR_BASE + 'figures/sims/' + str(fig_name) + '.png')
        
        plt.show()
    
    if save:      
      #import os
      #tmp_dir = os.getcwd() + "/" + DIR_BASE + "figures/sims/"
      comm = "convert -delay 10 -loop 0 "
      comm += DIR_BASE + "figures/sims/*.png "
      comm += DIR_BASE + "figures/sims/" + str(sec) + ".gif"
      #subprocess.call(comm)
      a = os.system(comm)
      #a = os.system("convert -delay 10 -loop 0 *.png " + str(sec) +".gif")
      print(a)
    
  def plotExperiment(self, U, B, per, directory=""):
    if per == 0: return
    
    X, Y = np.meshgrid(self.x, self.y)
    Xv, Yv = np.mgrid[self.x[0]:self.x[-1]:complex(0, self.N // 4), 
                      self.y[0]:self.y[-1]:complex(0, self.M // 4)]
    #fine = np.linspace(self.x[0], self.x[-1], 2*self.N)
    size = len(U)
    step = int(size / int(per*size))
    
    for i in range(0, size, step):
      plt.figure(figsize=(12, 8))
      
      plt.subplot(1, 2, 1)
      #fu = interp2d(self.x, self.y, U[i], kind='cubic')
      #UU = fu(fine, fine)  
      UU = U[i]
      im = plt.imshow(UU, origin='lower', cmap=plt.cm.jet, alpha=0.9, 
                 extent=[self.x[0], self.x[-1], self.y[0], self.y[-1]])
      plt.colorbar(im, fraction=0.046, pad=0.04)
      plt.quiver(Xv, Yv, self.v[0](Xv, Yv), self.v[1](Xv, Yv))  
      plt.xlabel("x")
      plt.ylabel("y")
      
      plt.subplot(1, 2, 2)
      #fb = interp2d(self.x, self.y, B[i], kind='cubic')
      #BB = fb(fine, fine)  
      BB = B[i]
      im2 = plt.imshow(BB, origin='lower', cmap=plt.cm.Oranges, alpha=1, 
                 extent=[self.x[0], self.x[-1], self.y[0], self.y[-1]])
      plt.colorbar(im2, fraction=0.046, pad=0.04)
      plt.xlabel("x")
      plt.ylabel("y")
      
      
      plt.tight_layout()
      
      if directory != "":
        if i // 10 < 10:
          fig_name = '0' + str(i // 10)
        else:
          fig_name = str(i // 10)        

        pathlib.Path(directory + "figures/").mkdir(parents=True, exist_ok=True) 
        plt.savefig(directory + "figures/" + fig_name + '.png', dpi=200)
      
      
      plt.show()

    
  def save(self, U, B, per=0):
    sec = int(datetime.today().timestamp())
    directory = "simulation/" + str(sec) + "/"

    pathlib.Path(directory).mkdir(parents=True, exist_ok=True) 
    np.save(directory + "U.npy", U)
    np.save(directory + "B.npy", B)
    
    parameters = {
      'u0': inspect.getsourcelines(self.u0)[0][0].strip("['\n']"),
      'beta0': inspect.getsourcelines(self.beta0)[0][0].strip("['\n']"),
      'kappa': self.kappa,
      'epsilon': self.epsilon,
      'upc': self.upc,
      'q': self.q,
      'v': (inspect.getsourcelines(self.v[0])[0][0].strip("['\n']"),
            inspect.getsourcelines(self.v[1])[0][0].strip("['\n']")),
      'alpha': self.alpha,
      'x': (self.x[0], self.x[-1], self.N),
      'y': (self.y[0], self.y[-1], self.M),
      't': (self.t[0], self.t[-1], self.T)
    }
    
    with open(directory + 'parameters.json', 'w') as fp:
      json.dump(parameters, fp)
    
    self.plotExperiment(U, B, per, directory)
    
    
    