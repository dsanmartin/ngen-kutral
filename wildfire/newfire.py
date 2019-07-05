import os
os.environ['MKL_NUM_THREADS'] = '1'
import numpy as np
from datetime import datetime
from wildfire.newdiffmat import FD1Matrix, FD2Matrix


sec = int(datetime.today().timestamp())
DIR_BASE = "simulation/" + str(sec) + "/"


class Fire:
  
  def __init__(self, parameters):
    # Physical Model parameters
    self.kap = parameters['kap']
    self.eps = parameters['eps']
    self.upc = parameters['upc']
    self.alp = parameters['alp'] 
    self.q = parameters['q']
    
    # Domain limits
    self.x_min, self.x_max = parameters['x_lim']
    self.y_min, self.y_max = parameters['y_lim']
    self.t_min, self.t_max = parameters['t_lim']
    
    # Domain resolution
    self.M = parameters['M']
    self.N = parameters['N']
    self.L = parameters['L']
    
    # Space-time domain 
    self.x = np.linspace(self.x_min, self.x_max, self.N)
    self.y = np.linspace(self.y_min, self.y_max, self.M)
    self.t = np.linspace(self.t_min, self.t_max, self.L)
    self.dx = self.x[1] - self.x[0]
    self.dy = self.y[1] - self.y[0]
    self.dt = self.t[1] - self.t[0]
    
    # Grid
    self.X, self.Y = np.meshgrid(self.x, self.y)
    
    # Numerical Methods
    self.space = parameters['space']
    self.time = parameters['time']
    
    # Initial conditions
    self.u0 = parameters['u0']
    self.b0 = parameters['b0']
    
    # Vector field
    self.v = parameters['v']
    
    # Others
    self.sparse = parameters['sparse']
    self.show = parameters['show']
    self.complete = parameters['complete']
    self.cmp = parameters['components']
    
    # Spatial
    if self.space == "FD":
      
      # RHS
      self.RHS = self.RHSFD
      
      # Differentiation matrices
      self.Dx = FD1Matrix(self.N, self.dx, self.sparse)
      self.Dy = FD1Matrix(self.M, self.dy, self.sparse)
      self.D2x = FD2Matrix(self.N, self.dx, self.sparse)
      self.D2y = FD2Matrix(self.M, self.dy, self.sparse)
      
    elif self.space == "FFT":
      x_len = self.x_max - self.x_min
      y_len = self.y_max - self.y_min
      
      self.x = np.linspace(self.x_min, self.x_max, self.N, endpoint=True)
      self.y = np.linspace(self.y_min, self.y_max, self.M, endpoint=True)
      
      self.X, self.Y = np.meshgrid(self.x, self.y)
      
      # Fourier domain
      eta = 2 * np.pi / y_len * np.fft.fftfreq(self.M, d=1/self.M)
      xi = 2 * np.pi / x_len * np.fft.fftfreq(self.N, d=1/self.N)
      self.XI, self.ETA = np.meshgrid(xi, eta)
      
      # RHS 
      self.RHS = self.RHSFFT
      
    
  
  def solvePDE(self, keep="all"):
    """
    Solve PDE model
    
    Parameters
    ----------
    keep : string
        * 'all' Keep and return all aproximations
        * 'last' Only keep and return last approximation 
            
    Returns
    -------
    U : (L, M, N) ndarray
      Temperatures approximation
    B : (L, M, N) ndarray  
      Fuels approximation
    """
    
    U0 = self.u0(self.X, self.Y)
    B0 = self.b0(self.X, self.Y)
    
    y0 = np.zeros((2 * self.M * self.N))
    
    y0[:self.M * self.N] = U0.flatten()
    y0[self.M * self.N:] = B0.flatten()
    
    if keep == "all":      
      if self.time == "Euler": integration = self.Euler
      elif self.time == "RK4": integration = self.RK4
    elif keep == "last":
      if self.time == "Euler": integration = self.EulerLast
      elif self.time == "RK4": integration = self.RK4Last
      
    # Integration
    y = integration(self.RHS, y0)
    
    if keep == "all":
      U = y[:, :self.M * self.N].reshape(self.L, self.M, self.N)
      B = y[:, self.M * self.N:].reshape(self.L, self.M, self.N)
    elif keep == "last":
      U = y[:self.M * self.N].reshape(self.M, self.N)
      B = y[self.M * self.N:].reshape(self.M, self.N)
      
    return U, B 
    
        
              
  def RHSFD(self, y, t, args=None):
    """
    Compute right hand side of PDE
    """
    # Vector field evaluation
    V1 = self.v[0](self.X, self.Y, t)
    V2 = self.v[1](self.X, self.Y, t)
    
    Uf = np.copy(y[:self.M * self.N].reshape((self.M, self.N)))
    Bf = np.copy(y[self.M * self.N:].reshape((self.M, self.N)))
    U = Uf
    B = Bf

    # Compute gradient of U, grad(U) = (U_x, U_y)
    if self.sparse:
      Ux, Uy = (self.Dx.dot(U.T)).T, self.Dy.dot(U)
      Uxx, Uyy = (self.D2x.dot(U.T)).T, self.D2y.dot(U)
    else:
      Ux, Uy = np.dot(U, self.Dx.T), np.dot(self.Dy, U)
      Uxx, Uyy = np.dot(U, self.D2x.T), np.dot(self.D2y, U)
      
    lapU = Uxx + Uyy
    
    if self.complete:
      K = self.K(U) 
      Kx = self.Ku(U) * Ux #(Dx.dot(K.T)).T
      Ky = self.Ku(U) * Uy #Dy.dot(K)
      diffusion = Kx * Ux + Ky * Uy + K * lapU
    else:
      diffusion = self.kap * lapU # k \nabla U
    
    convection = Ux * V1 + Uy * V2 # v \cdot grad u.    
    fuel = self.f(U, B) # eval fuel
    
    G = self.g(U, B)
    
    Uf = self.cmp[0] * diffusion - self.cmp[1] * convection + self.cmp[2] * fuel #diffusion - convection #+ fuel    
    Bf = G
    
    # Boundary conditions
    Uf[0,:] = np.zeros(self.N)
    Uf[-1,:] = np.zeros(self.N)
    Uf[:,0] = np.zeros(self.M)
    Uf[:,-1] = np.zeros(self.M)
    
    Bf[0,:] = np.zeros(self.N)
    Bf[-1,:] = np.zeros(self.N)
    Bf[:,0] = np.zeros(self.M)
    Bf[:,-1] = np.zeros(self.M)
    
    return np.r_[Uf.flatten(), Bf.flatten()]
  
  def RHSFFT(self, y, t):
    """
    Compute right hand side of PDE
    """
    # Vector field evaluation
    V1 = self.v[0](self.X, self.Y, t)
    V2 = self.v[1](self.X, self.Y, t)
    
    Uf = np.copy(y[:self.M * self.N].reshape((self.M, self.N)))
    Bf = np.copy(y[self.M * self.N:].reshape((self.M, self.N)))
    U = Uf
    B = Bf

    # Fourier transform
    Uhat = np.fft.fft2(U)
    
    # First derivative approximation
    Uhatx = 1j * self.XI * Uhat
    Uhaty = 1j * self.ETA * Uhat
    Uhatx[:,self.N//2] = np.zeros(self.M)
    Uhaty[self.M//2,:] = np.zeros(self.N)
    
    Ux = np.real(np.fft.ifft2(Uhatx))
    Uy = np.real(np.fft.ifft2(Uhaty))
    
    lap = np.real(np.fft.ifft2((-1 * (self.XI ** 2 + self.ETA ** 2)) * Uhat)) # Laplace operator
    diffusion = self.kap * lap # k \nabla U
    
    convection = Ux * V1 + Uy * V2 # v \cdot grad u.    
    fuel = self.f(U, B) # eval fuel
    
    G = self.g(U, B)
    
    Uf = self.cmp[0] * diffusion - self.cmp[1] * convection + self.cmp[2] * fuel #diffusion - convection #+ fuel    
    Bf = G
    
    # Boundary condition
    Uf[0,:] = np.zeros(self.N)
    Uf[-1,:] = np.zeros(self.N)
    Uf[:,0] = np.zeros(self.M)
    Uf[:,-1] = np.zeros(self.M)
    
    Bf[0,:] = np.zeros(self.N)
    Bf[-1,:] = np.zeros(self.N)
    Bf[:,0] = np.zeros(self.M)
    Bf[:,-1] = np.zeros(self.M)
    
    return np.r_[Uf.flatten(), Bf.flatten()]#np.concatenate((Uf.flatten(), Bf.flatten()))
  
  
  # TIME NUMERICAL METHODS #
  # Euler method
  def Euler(self, F, y0):
    
    y = np.zeros((self.L, 2 * self. M * self. N))
    y[0] = y0
    
    for k in range(self.L - 1):
      y[k+1] = y[k] + self.dt * F(y[k], self.t[k])
      
    return y
  
  # Runge-Kutta 4th order 
  def RK4(self, F, y0):
    
    y = np.zeros((self.L, 2 * self. M * self. N))
    y[0] = y0
    
    for k in range(self.L - 1):
      k1 = F(y[k], self.t[k])
      k2 = F(y[k] + 0.5 * self.dt * k1, self.t[k])
      k3 = F(y[k] + 0.5 * self.dt * k2, self.t[k])
      k4 = F(y[k] + self.dt * k3, self.t[k])

      y[k + 1] = y[k] + (1/6) * self.dt * (k1 + 2 * k2 + 2 * k3 + k4)
      
    return y
  
  # Runge-Kutta 4th order for time 
  def RK4Last(self, F, y0):
    y = y0
    
    for k in range(self.L - 1):
      yc = np.copy(y)
      k1 = F(yc, self.t[k])
      k2 = F(yc + 0.5 * self.dt * k1, self.t[k])
      k3 = F(yc + 0.5 * self.dt * k2, self.t[k])
      k4 = F(yc + self.dt * k3, self.t[k])

      y = yc + (1/6) * self.dt * (k1 + 2*k2 + 2*k3 + k4)
      
    return y
  
  def EulerLast(self, F, y0):    
    y = y0
    
    for k in range(self.L - 1):
      yc = np.copy(y)
      y = yc + self.dt * F(yc, self.t[k])
      
    return y
  
  # Functions #
  def K(self, u):
    return self.kap * (1 + self.eps * u) ** 3 + 1
  
  def Ku(self, u):
    return 3 * self.eps * self.kap * (1 + self.eps * u) ** 2
  
  def f(self, u, b):
    return self.s(u) * b * np.exp(u / (1 + self.eps * u)) - self.alp * u
  
  def g(self, u, b):
    return -self.s(u) * (self.eps / self.q) * b * np.exp(u /(1 + self.eps * u))
    
  def s(self, u):
    S = np.zeros_like(u)
    S[u >= self.upc] = 1
    return S
  