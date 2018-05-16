import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint
from scipy.interpolate import interp2d

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
    self.dt = parameters['dt']
    self.T = parameters['T']
    self.M, self.N = self.u0.shape
    self.x = np.linspace(0, 1, self.M)
    self.y = np.linspace(0, 1, self.N)
    self.t = np.linspace(0, self.dt * self.T, self.T)
    self.dx = self.x[1] - self.x[0]
    self.dy = self.y[1] - self.y[0]
    self.dt = self.t[1] - self.t[0]

    
  def div(self, v, u):
    X, Y = np.meshgrid(self.x, self.y)
    
    gradu = self.gradient(u)
    
    v1 = v[0](X, Y)
    v2 = v[1](X, Y)
    
    dv1 = self.derivative(v1, 0)
    dv2 = self.derivative(v2, 1)
    
    return np.dot(gradu[0], v1) + np.dot(gradu[1], v2)+ np.dot(u, dv1) + np.dot(u, dv2)
    #return gradu[0]*v1 + gradu[1]*v2 + u*dv1 + u*dv2
    
  def gradient(self, f):
    #dudx = (np.roll(u, -1, axis=0) + np.roll(u, 1, axis=0) - 2*u) / self.dx
    #dudy = (np.roll(u, -1, axis=1) + np.roll(u, 1, axis=1) - 2*u) / self.dy
    
    dfdx = self.derivative(f, 0)
    dfdy = self.derivative(f, 1)
    
    #dudx = np.gradient(u, self.dx, axis=0)#+ \
    #dudy = np.gradient(u, self.dy, axis=1)
    
    return (dfdx, dfdy)
  
  def derivative(self, f, axis_):
    h = 0
    if axis_ == 0:
      h = self.dx
    elif axis_ == 1:
      h = self.dy
      
    return (np.roll(f, -1, axis=axis_) + np.roll(f, 1, axis=axis_) - 2*f) / h
  
  def laplacian(self, u):
    return (np.roll(u, 1,axis=0) + np.roll(u, -1, axis=0) + \
              np.roll(u, -1,axis=1) + np.roll(u, 1, axis=1) - 4*u) / self.dx ** 2
            
  def F(self, u, beta):    
    
    W = np.zeros_like(u)
    
    diffusion = self.kappa * self.laplacian(u)
    convection = self.div(self.v, u)
    fuel = self.f(u, beta)
    
    #print("dif", np.max(diffusion))
    #print("conv", np.max(convection))
    #print("fuel", np.max(fuel))
    
    W = diffusion - convection #+ fuel
    
    #if np.isnan(np.max(W)):
    #  return
        
    return W
  
  def K(self, u):
    return self.kappa * (1 + self.epsilon * u) ** 3 + 1
  
  def f(self, u, beta):
    return self.s(u) * beta * np.exp(u /(1 + self.epsilon)) - self.alpha * u
  
  def g(self, u, beta):
    return -self.s(u) * (self.epsilon / self.q) * beta * np.exp(u /(1 + self.epsilon))
    
  def s(self, u):
    S = np.zeros_like(u)
    S[u >= self.upc] = 1
    
    return S
    
    
    
  # Solve PDE
  def solvePDE(self):
            
    U = np.zeros((self.T+1, self.M, self.N))
    B = np.zeros((self.T+1, self.M, self.N))
    
    U[0] = self.u0
    B[0] = self.beta0
        
    for t in range(1, self.T + 1):

      U[t] = U[t-1] + self.F(U[t-1], B[t-1]) * self.dt
      B[t] = B[t-1] + self.g(U[t-1], B[t-1]) * self.dt
        
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
    fine = np.linspace(0, 1, 2*self.N)
    fu = interp2d(self.x, self.y, temperatures[t], kind='cubic')
    U = fu(fine, fine)
    #U = temperatures[t].reshape(self.u0.shape)
    plt.imshow(U, origin='lower', cmap=plt.cm.jet)
    plt.colorbar()
    plt.show()