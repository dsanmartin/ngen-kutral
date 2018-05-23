import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint
from scipy.interpolate import interp2d
import pathlib, json, inspect, os
from datetime import datetime

sec = int(datetime.today().timestamp())
DIR_BASE = "simulation/" + str(sec) + "/"

# Chebyshev differentiation matrix
def cheb(N):
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
    self.M, self.N = len(self.x), len(self.y)
    self.T = len(self.t)
    self.dx = self.x[1] - self.x[0]
    self.dy = self.y[1] - self.y[0]
    self.dt = self.t[1] - self.t[0]

  def divergence(self, F):
    f1, f2 = F
    df1dx = self.derivative(f1, 0)
    df2dy = self.derivative(f2, 1)
    
    return df1dx + df2dy
    
    
  def conv(self, v, u):
    X, Y = np.meshgrid(self.x, self.y)
    
    gradu = self.gradient(u)
    
    v1 = v[0](X, Y)
    v2 = v[1](X, Y)
    
    #dv1 = self.derivative(v1, 0)
    #dv2 = self.derivative(v2, 1)
    
#    dv1 = np.gradient(v1, self.dx, axis=0)
#    dv2 = np.gradient(v2, self.dy, axis=1)
#    
#    gradux = np.gradient(u, self.dx, axis=0)
#    graduy = np.gradient(u, self.dy, axis=1)
    
    
    dv1 = np.gradient(v1, self.dx, axis=1)
    dv2 = np.gradient(v2, self.dy, axis=0)
    
    gradux = np.gradient(u, self.dx, axis=1)
    graduy = np.gradient(u, self.dy, axis=0) 
    
    #print(np.linalg.norm(gradux-gradu[0]))
    #print(np.linalg.norm(graduy-gradu[1]))
    
    #return np.dot(gradu[0], v1) + np.dot(v2, gradu[1])# + u*dv1 + u*dv2
    #return np.dot(gradu[0], v1) + np.dot(gradu[1], v2) + np.dot(u, dv1) + np.dot(u, dv2)
    #return np.dot(v1, gradu[0]) + np.dot(gradu[1], v2) + np.dot(dv1, u) + np.dot(u, dv2) 
    #return np.dot(gradux, v1) + np.dot(graduy, v2) + np.dot(u, dv1) + np.dot(u, dv2) 
    #return gradux*v1 + v2*graduy
    #return np.dot(gradux, v1) + np.dot(v2, graduy) + np.dot(u, dv1) + np.dot(dv2, u) 
    #return np.dot(gradux, v1) + np.dot(graduy, v2) + np.dot(u, dv1) + np.dot(u, dv2) 
    #return np.dot(gradu[0], v1) + np.dot(gradu[1], v2) + u*dv1 + u*dv2  
    #return np.dot(gradu[0], v1) + np.dot(gradu[1], v2) + np.dot(u, dv1) + np.dot(dv2, u)  
    
    #divV = self.divergence((v1, v2))
    #return u * divV + gradu[0]*v1 + gradu[1]*v2
    #return u * divV + ) + np.dot(v2, graduy)
    #return np.dot(u, dv1) + np.dot(u, dv2) + np.dot(v1, gradux) + np.dot(v2, graduy)
    #return np.dot(v1, gradux) + np.dot(graduy, v2)
    #return gradu[0]*v1 + gradu[1]*v2 + u*dv1 + u*dv2
    
    return gradux*v1 + u*dv1 + graduy*v2 + u*dv2
    #return v1*gradux + dv1*u + v2*graduy + dv2*u
    #return np.dot(u, dv1) + np.dot(dv2, u) + v1*gradux + v2*graduy
    
    
    
  def gradient(self, f):
    
    #dfdx = self.derivative(f, 0)
    #dfdy = self.derivative(f, 1)
    
    dfdx = np.gradient(f, self.dx, axis=0)
    dfdy = np.gradient(f, self.dy, axis=1)
    
    return (dfdx, dfdy)
  
  def derivative(self, f, axis_):
    h = 0
    if axis_ == 0:
      h = self.dx
    elif axis_ == 1:
      h = self.dy
      
    return (np.roll(f, -1, axis=axis_) - np.roll(f, 1, axis=axis_)) / (2*h)
  
  def laplacian(self, u):
    return (np.roll(u, 1, axis=0) + np.roll(u, -1, axis=0) + \
              np.roll(u, -1,axis=1) + np.roll(u, 1, axis=1) - 4*u) / self.dx ** 2
            
  def F(self, u, beta):    
    
    W = np.zeros_like(u)
    
    diffusion = (self.kappa * self.laplacian(u))
    convection = self.conv(self.v, u)
    fuel = self.f(u, beta)
        
#    mdif = np.max(diffusion)
#    mcon = np.max(convection)
#    mfue = np.max(fuel)
#    
#    midif = np.min(diffusion)
#    micon = np.min(convection)
#    mifue = np.min(fuel)
#    
#    print("dif", mdif)
#    print("conv", mcon)
#    print("fuel", mfue)
#    
#    if np.isnan(mdif) or np.isnan(mcon) or np.isnan(mfue) or np.isnan(midif) or np.isnan(micon) or np.isnan(mifue):
#      return

    #W = diffusion - convection #+ beta*u #+ fuel
    #W = -convection
    W = diffusion - convection + fuel
    
    W[0,:] = np.zeros(self.N)
    W[-1,:] = np.zeros(self.N)
    W[:,0] = np.zeros(self.M)
    W[:,-1] = np.zeros(self.M)
        
    return W
  
  def Fcheb(self, W, t, mu, A, V1, V2, Dx, Dy):
    D2x = np.dot(Dx, Dx)
    D2y = np.dot(Dy, Dy)
    
    N = Dx.shape[0]
    
    # Reshape W to Matrix
    W = W.reshape(N, N)
    diff = mu*(np.dot(W, D2x.T) + np.dot(D2y, W))    
    conv = np.dot(np.dot(W, Dx.T), V1) + np.dot(W, np.dot(V1, Dx.T)) \
        + np.dot(np.dot(Dy, W), V2) + np.dot(W, np.dot(Dy, V2))
    reac = np.dot(A, W)
    
    W = diff - conv #+ reac
    
    # Boundary conditions
    W[0,:] = np.zeros(N)
    W[-1,:] = np.zeros(N)
    W[:,0] = np.zeros(N)
    W[:,-1] = np.zeros(N)
    
    return W.flatten() # Flatten for odeint
  
  def K(self, u):
    return self.kappa * (1 + self.epsilon * u) ** 3 + 1
  
  def f(self, u, beta):
    return self.s(u) * beta * np.exp(u /(1 + self.epsilon*u)) - self.alpha * u
  
  def g(self, u, beta):
    return -self.s(u) * (self.epsilon / self.q) * beta * np.exp(u /(1 + self.epsilon*u))
    
  def s(self, u):
    S = np.zeros_like(u)
    S[u >= self.upc] = 1
    
    return S
  
  
  def solveRK4(self, U0, B0):
    U = np.zeros((self.T+1, self.M, self.N))
    B = np.zeros((self.T+1, self.M, self.N))
    
    U[0] = U0
    B[0] = B0
    
    for t in range(1, self.T + 1):
      k1 = self.F(U[t-1], B[t-1])
      k2 = self.F(U[t-1] + 0.5*self.dt*k1, B[t-1] + 0.5*self.dt*k1)
      k3 = self.F(U[t-1] + 0.5*self.dt*k2, B[t-1] + 0.5*self.dt*k2)
      k4 = self.F(U[t-1] + self.dt*k3, B[t-1] + self.dt*k3)

      U[t] = U[t-1] + (1/6)*self.dt*(k1 + 2*k2 + 2*k3 + k4)
      
      bk1 = self.g(U[t-1], B[t-1])
      bk2 = self.g(U[t-1] + 0.5*self.dt*bk1, B[t-1] + 0.5*self.dt*bk1)
      bk3 = self.g(U[t-1] + 0.5*self.dt*bk2, B[t-1] + 0.5*self.dt*bk2)
      bk4 = self.g(U[t-1] + self.dt*bk3, B[t-1] + self.dt*bk3)

      B[t] = B[t-1] + (1/6)*self.dt*(bk1 + 2*bk2 + 2*bk3 + bk4)
      
      U[t,0,:] = np.zeros(self.N)
      U[t,-1,:] = np.zeros(self.N)
      U[t,:,0] = np.zeros(self.N)
      U[t,:,-1] = np.zeros(self.N)
      
    return U, B
      
        
  def solveEuler(self, U0, B0):

    U = np.zeros((self.T+1, self.M, self.N))
    B = np.zeros((self.T+1, self.M, self.N))
    
    U[0] = U0
    B[0] = B0
    
    for t in range(1, self.T + 1):
      U[t] = U[t-1] + self.F(U[t-1], B[t-1]) * self.dt
      B[t] = B[t-1] + self.g(U[t-1], B[t-1]) * self.dt
      
      U[t,0,:] = np.zeros(self.N)
      U[t,-1,:] = np.zeros(self.N)
      U[t,:,0] = np.zeros(self.M)
      U[t,:,-1] = np.zeros(self.M)
      
    return U, B
      
  # Solve PDE with cheb
  def solvePDECheb(self):
    Dx, x = cheb(self.M)
    Dy, y = cheb(self.N)
    
    X, Y = np.meshgrid(x, y)
    V1 = self.v[0](X, Y)
    V2 = self.v[1](X, Y)
    
    A = self.beta0(X, Y)
    W = self.u0(X, Y)
    
    W = odeint(self.Fcheb, W.flatten(), self.t, 
               args=(self.kappa, A, V1, V2, Dx, Dy))
    U = []
    for w in W:
      U.append(w.reshape(self.M + 1, self.N + 1 ))
          
    return np.array(U)   
    
  # Solve PDE
  def solvePDE(self, method='rk4'):
            
    #U = np.zeros((self.T+1, self.M, self.N))
    #B = np.zeros((self.T+1, self.M, self.N))
    
    X, Y = np.meshgrid(self.x, self.y)
    
    #U[0] = self.u0(X, Y)
    #B[0] = self.beta0(X, Y)
    U0 = self.u0(X, Y)
    B0 = self.beta0(X, Y)
    
    if method == 'rk4':
      U, B = self.solveRK4(U0, B0)
    elif method == 'euler': 
      U, B = self.solveEuler(U0, B0)
    elif method == 'cheb':
      U = self.solvePDECheb()
      B = np.zeros_like(U)
        
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
    _, x = cheb(N-1)
    _, y = cheb(N-1)
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
    
    Xv, Yv = np.mgrid[self.x[0]:self.x[-1]:complex(0, self.M // 2), 
                      self.y[0]:self.y[-1]:complex(0, self.N // 2)]
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
    
  def plots(self, U, B, save=False):
    
    sec = int(datetime.today().timestamp())
    DIR_BASE = "simulation/" + str(sec) + "/"
    
    for i in range(self.T):
      if i % 10 == 0:
        plt.subplot(1, 2, 1)
        plt.imshow(U[i], origin='lower', cmap=plt.cm.jet, alpha=0.9, vmin=np.min(U),
                   vmax=np.max(U), extent=[self.x[0], self.x[-1], self.y[0], self.y[-1]])
        plt.colorbar(fraction=0.046, pad=0.04)
        
        Xv, Yv = np.mgrid[self.x[0]:self.x[-1]:complex(0, self.M // 2), 
                          self.y[0]:self.y[-1]:complex(0, self.N // 2)]
        plt.quiver(Xv, Yv, self.v[0](Xv, Yv), self.v[1](Xv, Yv)) 
        
        plt.title("Temperature + Wind")
        plt.xlabel("x")
        plt.ylabel("y")
        
        plt.subplot(1, 2, 2)
        plt.imshow(B[i], origin='lower', cmap=plt.cm.Oranges, alpha=1, vmin=np.min(B),
                   vmax=np.max(B), extent=[self.x[0], self.x[-1], self.y[0], self.y[-1]])
        plt.colorbar(fraction=0.046, pad=0.04)  
        plt.title("Fuel")
        plt.xlabel("x")
        plt.ylabel("y")
        
        plt.tight_layout()
        
        if save:
          fig_n = i // 10 + 1
          if fig_n < 10:
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
    Xv, Yv = np.mgrid[self.x[0]:self.x[-1]:complex(0, self.M // 4), 
                      self.y[0]:self.y[-1]:complex(0, self.N // 4)]
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
    
    
    