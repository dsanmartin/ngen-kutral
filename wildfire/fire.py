import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
from scipy import optimize
from wildfire.diffmat import FD1Matrix, FD2Matrix, chebyshevMatrix


sec = int(datetime.today().timestamp())
DIR_BASE = "simulation/" + str(sec) + "/"


class Fire:
  
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
    self.sparse = parameters['sparse']
    self.show = parameters['show']
    self.complete = parameters['complete']
    self.components = parameters['components']
        
            
  # RHS of PDE
  def RHS(self, U, B, V, args=None):
    """
    Compute right hand side of PDE
    """
    Dx, Dy, D2x, D2y = args # Unpack differentiation matrices
    V1, V2 = V # Unpack tuple of Vector field

    # Compute gradient of U, grad(U) = (U_x, U_y)
    if self.sparse:
      Ux, Uy = (Dx.dot(U.T)).T, Dy.dot(U)
    else:
      Ux, Uy = np.dot(U, Dx.T), np.dot(Dy, U)
    
    # Compute laplacian of U, div(grad U) = Uxx + Uyy
    if self.sparse:
      Uxx, Uyy = (D2x.dot(U.T)).T, D2y.dot(U)
    else:
      Uxx, Uyy = np.dot(U, D2x.T), np.dot(D2y, U)
      
    lapU = Uxx + Uyy
    
    if self.complete:
      K = self.K(U) 
      Kx = self.Ku(U) * Ux #(Dx.dot(K.T)).T
      Ky = self.Ku(U) * Uy #Dy.dot(K)
      diffusion = Kx * Ux + Ky * Uy + K * lapU
    else:
      diffusion = self.kappa * lapU # k \nabla U
    
    convection = Ux * V1 + Uy * V2 # v \cdot grad u.    
    fuel = self.f(U, B) # eval fuel
    
    return self.components[0]*diffusion - self.components[1]*convection + self.components[2]*fuel    
    
  def RHSvec(self, y, V, args=None):
    """
    Compute right hand side of PDE
    """
    Dx, Dy, D2x, D2y = args # Unpack differentiation matrices
    V1, V2 = V # Unpack tuple of Vector field
    
    Uf = np.copy(y[:self.M * self.N].reshape((self.M, self.N)))
    Bf = np.copy(y[self.M * self.N:].reshape((self.M, self.N)))
    U = Uf[1:-1, 1:-1]
    B = Bf[1:-1, 1:-1]

    # Compute gradient of U, grad(U) = (U_x, U_y)
    if self.sparse:
      Ux, Uy = (Dx.dot(U.T)).T, Dy.dot(U)
      Uxx, Uyy = (D2x.dot(U.T)).T, D2y.dot(U)
    else:
      Ux, Uy = np.dot(U, Dx.T), np.dot(Dy, U)
      Uxx, Uyy = np.dot(U, D2x.T), np.dot(D2y, U)
      
    lapU = Uxx + Uyy
    
    if self.complete:
      K = self.K(U) 
      Kx = self.Ku(U) * Ux #(Dx.dot(K.T)).T
      Ky = self.Ku(U) * Uy #Dy.dot(K)
      diffusion = Kx * Ux + Ky * Uy + K * lapU
    else:
      diffusion = self.kappa * lapU # k \nabla U
    
    convection = Ux * V1 + Uy * V2 # v \cdot grad u.    
    fuel = self.f(U, B) # eval fuel
    
    G = self.g(U, B)
    
    Uf[1:-1, 1:-1] = self.components[0]*diffusion - self.components[1]*convection + self.components[2]*fuel #diffusion - convection #+ fuel    
    Bf[1:-1, 1:-1] = G
    
    Uf[0,:] = np.zeros(self.N)
    Uf[-1,:] = np.zeros(self.N)
    Uf[:,0] = np.zeros(self.M)
    Uf[:,-1] = np.zeros(self.M)
    
    Bf[0,:] = np.zeros(self.N)
    Bf[-1,:] = np.zeros(self.N)
    Bf[:,0] = np.zeros(self.M)
    Bf[:,-1] = np.zeros(self.M)
    
    return np.r_[Uf.flatten(), Bf.flatten()]#np.concatenate((Uf.flatten(), Bf.flatten()))
    
    
  def K(self, u):
    return self.kappa * (1 + self.epsilon * u) ** 3 + 1
  
  def Ku(self, u):
    return 3 * self.epsilon * self.kappa * (1 + self.epsilon * u) ** 2
  
  def f(self, u, beta):
    return self.s(u) * beta * np.exp(u / (1 + self.epsilon*u)) - self.alpha * u
  
  def g(self, u, beta):
    return -self.s(u) * (self.epsilon / self.q) * beta * np.exp(u /(1 + self.epsilon*u))
    
  def s(self, u):
    S = np.zeros_like(u)
    S[u >= self.upc] = 1
    
    return S
  
  # Runge-Kutta 4th order for time 
  def solveRK4vec(self, U0, B0, V, args):
    M, N = U0.shape
    
    y = np.zeros((self.T, 2 * M * N))
    
    y[0, :M * N] = U0.flatten()
    y[0, M * N:] = B0.flatten()
    
    X, Y = np.meshgrid(self.x, self.y)
    
    for t in range(1, self.T):
      V1 = self.v[0](X, Y, self.t[t])
      V2 = self.v[1](X, Y, self.t[t])

      V = (V1[1:-1, 1:-1], V2[1:-1, 1:-1]) # Vector field
      
      k1 = self.RHSvec(y[t-1], V, args)
      k2 = self.RHSvec(y[t-1] + 0.5*self.dt*k1, V, args)
      k3 = self.RHSvec(y[t-1] + 0.5*self.dt*k2, V, args)
      k4 = self.RHSvec(y[t-1] + self.dt*k3, V, args)

      y[t] = y[t-1] + (1/6) * self.dt * (k1 + 2*k2 + 2*k3 + k4)
      
    return y[:, :M * N].reshape(self.T, M, N), y[:, M * N:].reshape(self.T, M, N)
  
  # Runge-Kutta 4th order for time 
  def solveRK4vecLast(self, U0, B0, V, args):
    M, N = U0.shape
    
    y = np.zeros((2 * M * N))
    
    y[:M * N] = U0.flatten()
    y[M * N:] = B0.flatten()
    
    X, Y = np.meshgrid(self.x, self.y)
    
    for t in range(1, self.T):
      V1 = self.v[0](X, Y, self.t[t])
      V2 = self.v[1](X, Y, self.t[t])

      V = (V1[1:-1, 1:-1], V2[1:-1, 1:-1]) # Vector field
      
      yc = np.copy(y)
      
      k1 = self.RHSvec(yc, V, args)
      k2 = self.RHSvec(yc + 0.5*self.dt*k1, V, args)
      k3 = self.RHSvec(yc + 0.5*self.dt*k2, V, args)
      k4 = self.RHSvec(yc + self.dt*k3, V, args)

      y = yc + (1/6) * self.dt * (k1 + 2*k2 + 2*k3 + k4)
      
    return y[:M * N].reshape(M, N), y[M * N:].reshape(M, N)
  
  def solveEulerVecLast(self, U0, B0, V, args):
    M, N = U0.shape
    
    y = np.zeros((2 * M * N))
    
    y[:M * N] = U0.flatten()
    y[M * N:] = B0.flatten()
    
    X, Y = np.meshgrid(self.x, self.y)
    
    for t in range(1, self.T):
      V1 = self.v[0](X, Y, self.t[t])
      V2 = self.v[1](X, Y, self.t[t])

      V = (V1[1:-1, 1:-1], V2[1:-1, 1:-1]) # Vector field
      
      yc = np.copy(y)
      
      yn = self.RHSvec(yc, V, args)

      y = yc + self.dt * yn
      
    return y[:M * N].reshape(M, N), y[M * N:].reshape(M, N)
  
  # Runge-Kutta 4th order for time 
  def solveODEIntLast(self, U0, B0, V, args):
    M, N = U0.shape
    
    y0 = np.zeros((2 * M * N))
    
    y0[:M * N] = U0.flatten()
    y0[M * N:] = B0.flatten()
    
    X, Y = np.meshgrid(self.x, self.y)
    
    from scipy.integrate import solve_ivp
    
    #for t in range(1, self.T):
    V1 = self.v[0](X, Y, self.t[0])
    V2 = self.v[1](X, Y, self.t[0])

    V = (V1[1:-1, 1:-1], V2[1:-1, 1:-1]) # Vector field
    
    y = solve_ivp(fun=lambda t, y: self.RHSvec(y, V, args), y0=y0, 
                  t_span=(self.t[0], self.t[-1]), max_step=self.dt, 
                  t_eval=[self.t[-1]], method='RK45')
      
    return y.y[:M * N].reshape(M, N), y.y[M * N:].reshape(M, N)
  
  # Runge-Kutta 4th order for time 
  def solveRK4(self, U0, B0, V, args):
    M, N = U0.shape
    
    U = np.zeros((self.T, M, N))
    B = np.zeros((self.T, M, N))
    
    U[0] = U0
    B[0] = B0
    
    X, Y = np.meshgrid(self.x, self.y)
    
    for t in range(1, self.T):
      V1 = self.v[0](X, Y, self.t[t])
      V2 = self.v[1](X, Y, self.t[t])

      V = (V1[1:-1, 1:-1], V2[1:-1, 1:-1]) # Vector field
      
      k1 = self.RHS(U[t-1, 1:-1, 1:-1], B[t-1, 1:-1, 1:-1], V, args)
      k2 = self.RHS(U[t-1, 1:-1, 1:-1] + 0.5*self.dt*k1, B[t-1, 1:-1, 1:-1] + 0.5*self.dt*k1, V, args)
      k3 = self.RHS(U[t-1, 1:-1, 1:-1] + 0.5*self.dt*k2, B[t-1, 1:-1, 1:-1] + 0.5*self.dt*k2, V, args)
      k4 = self.RHS(U[t-1, 1:-1, 1:-1] + self.dt*k3, B[t-1, 1:-1, 1:-1] + self.dt*k3, V, args)

      U[t, 1:-1, 1:-1] = U[t-1, 1:-1, 1:-1] + (1/6)*self.dt*(k1 + 2*k2 + 2*k3 + k4)
      
      # BC of temperature
      U[t,0,:] = np.zeros(N)
      U[t,-1,:] = np.zeros(N)
      U[t,:,0] = np.zeros(M)
      U[t,:,-1] = np.zeros(M)
      
      bk1 = self.g(U[t-1, 1:-1, 1:-1], B[t-1, 1:-1, 1:-1])
      bk2 = self.g(U[t-1, 1:-1, 1:-1] + 0.5*self.dt*bk1, B[t-1, 1:-1, 1:-1] + 0.5*self.dt*bk1)
      bk3 = self.g(U[t-1, 1:-1, 1:-1] + 0.5*self.dt*bk2, B[t-1, 1:-1, 1:-1] + 0.5*self.dt*bk2)
      bk4 = self.g(U[t-1, 1:-1, 1:-1] + self.dt*bk3, B[t-1, 1:-1, 1:-1] + self.dt*bk3)

      B[t, 1:-1, 1:-1] = B[t-1, 1:-1, 1:-1] + (1/6)*self.dt*(bk1 + 2*bk2 + 2*bk3 + bk4)
      
      # BC of fuel
      B[t,0,:] = np.zeros(N)
      B[t,-1,:] = np.zeros(N)
      B[t,:,0] = np.zeros(M)
      B[t,:,-1] = np.zeros(M)
      
    return U, B
  
  
  # Only save last value
  def solveRK4last(self, U0, B0, V, args):
    M, N = U0.shape
    
    U = U0
    B = B0
    
    X, Y = np.meshgrid(self.x, self.y)
    
    for t in range(1, self.T):
      V1 = self.v[0](X, Y, self.t[t])
      V2 = self.v[1](X, Y, self.t[t])

      V = (V1[1:-1, 1:-1], V2[1:-1, 1:-1]) # Vector field
      
      Uc = np.copy(U)
      Bc = np.copy(B)
      
      k1 = self.RHS(Uc[1:-1, 1:-1], Bc[1:-1, 1:-1], V, args)
      k2 = self.RHS(Uc[1:-1, 1:-1] + 0.5*self.dt*k1, Bc[1:-1, 1:-1] + 0.5*self.dt*k1, V, args)
      k3 = self.RHS(Uc[1:-1, 1:-1] + 0.5*self.dt*k2, Bc[1:-1, 1:-1] + 0.5*self.dt*k2, V, args)
      k4 = self.RHS(Uc[1:-1, 1:-1] + self.dt*k3, Bc[1:-1, 1:-1] + self.dt*k3, V, args)

      U[1:-1, 1:-1] = Uc[1:-1, 1:-1] + (1/6)*self.dt*(k1 + 2*k2 + 2*k3 + k4)
      
      # BC of temperature
      U[0,:] = np.zeros(N)
      U[-1,:] = np.zeros(N)
      U[:,0] = np.zeros(M)
      U[:,-1] = np.zeros(M)
      
      bk1 = self.g(Uc[1:-1, 1:-1], Bc[1:-1, 1:-1])
      bk2 = self.g(Uc[1:-1, 1:-1] + 0.5*self.dt*bk1, Bc[1:-1, 1:-1] + 0.5*self.dt*bk1)
      bk3 = self.g(Uc[1:-1, 1:-1] + 0.5*self.dt*bk2, Bc[1:-1, 1:-1] + 0.5*self.dt*bk2)
      bk4 = self.g(Uc[1:-1, 1:-1] + self.dt*bk3, Bc[1:-1, 1:-1] + self.dt*bk3)

      B[1:-1, 1:-1] = Bc[1:-1, 1:-1] + (1/6)*self.dt*(bk1 + 2*bk2 + 2*bk3 + bk4)
      
      # BC of fuel
      B[0,:] = np.zeros(N)
      B[-1,:] = np.zeros(N)
      B[:,0] = np.zeros(M)
      B[:,-1] = np.zeros(M)
      
    return U, B
  
  def solveRK4Data(self, U0, B0, V, args):
    M, N = U0.shape
    
    U = np.zeros((self.T, M, N))
    B = np.zeros((self.T, M, N))
    
    U[0] = U0
    B[0] = B0
    
    X, Y = np.meshgrid(self.x, self.y)
    
    for t in range(1, self.T):
      V1 = self.v[t-1, 0]
      V2 = self.v[t-1, 1]
      
      #print(V1, V2)

      #V = (V1[1:-1, 1:-1], V2[1:-1, 1:-1]) # Vector field
      V = (V1*np.ones((self.M-2, self.N-2)), V2*np.ones((self.M-2, self.N-2)))
      
      
      k1 = self.RHS(U[t-1, 1:-1, 1:-1], B[t-1, 1:-1, 1:-1], V, args)
      k2 = self.RHS(U[t-1, 1:-1, 1:-1] + 0.5*self.dt*k1, B[t-1, 1:-1, 1:-1] + 0.5*self.dt*k1, V, args)
      k3 = self.RHS(U[t-1, 1:-1, 1:-1] + 0.5*self.dt*k2, B[t-1, 1:-1, 1:-1] + 0.5*self.dt*k2, V, args)
      k4 = self.RHS(U[t-1, 1:-1, 1:-1] + self.dt*k3, B[t-1, 1:-1, 1:-1] + self.dt*k3, V, args)

      U[t, 1:-1, 1:-1] = U[t-1, 1:-1, 1:-1] + (1/6)*self.dt*(k1 + 2*k2 + 2*k3 + k4)
      
      # BC of temperature
      U[t,0,:] = np.zeros(N)
      U[t,-1,:] = np.zeros(N)
      U[t,:,0] = np.zeros(M)
      U[t,:,-1] = np.zeros(M)
      
      bk1 = self.g(U[t-1, 1:-1, 1:-1], B[t-1, 1:-1, 1:-1])
      bk2 = self.g(U[t-1, 1:-1, 1:-1] + 0.5*self.dt*bk1, B[t-1, 1:-1, 1:-1] + 0.5*self.dt*bk1)
      bk3 = self.g(U[t-1, 1:-1, 1:-1] + 0.5*self.dt*bk2, B[t-1, 1:-1, 1:-1] + 0.5*self.dt*bk2)
      bk4 = self.g(U[t-1, 1:-1, 1:-1] + self.dt*bk3, B[t-1, 1:-1, 1:-1] + self.dt*bk3)

      B[t, 1:-1, 1:-1] = B[t-1, 1:-1, 1:-1] + (1/6)*self.dt*(bk1 + 2*bk2 + 2*bk3 + bk4)
      
      # BC of fuel
      B[t,0,:] = np.zeros(N)
      B[t,-1,:] = np.zeros(N)
      B[t,:,0] = np.zeros(M)
      B[t,:,-1] = np.zeros(M)
      
    return U, B
      
  # Forward Euler for time
  def solveEuler(self, U0, B0, V, args):
    M, N = U0.shape
    
    U = np.zeros((self.T, M, N))
    B = np.zeros((self.T, M, N))
    
    U[0] = U0
    B[0] = B0
    
    X, Y = np.meshgrid(self.x, self.y)
    
    for t in range(1, self.T):
      V1 = self.v[0](X, Y, self.t[t])
      V2 = self.v[1](X, Y, self.t[t])

      V = (V1[1:-1, 1:-1], V2[1:-1, 1:-1]) # Vector field
      
      U[t, 1:-1, 1:-1] = U[t-1, 1:-1, 1:-1] + self.RHS(U[t-1, 1:-1, 1:-1], B[t-1, 1:-1, 1:-1], V, args) * self.dt
      B[t, 1:-1, 1:-1] = B[t-1, 1:-1, 1:-1] + self.g(U[t-1, 1:-1, 1:-1], B[t-1, 1:-1, 1:-1]) * self.dt
      
      U[t,0,:] = np.zeros(N)
      U[t,-1,:] = np.zeros(N)
      U[t,:,0] = np.zeros(M)
      U[t,:,-1] = np.zeros(M)
      
      B[t,0,:] = np.zeros(N)
      B[t,-1,:] = np.zeros(N)
      B[t,:,0] = np.zeros(M)
      B[t,:,-1] = np.zeros(M)
      
    return U, B
  
  
  def solveImpEuler(self, U0, B0, V, args):
    M, N = U0.shape
    
    U = np.zeros((self.T, M, N))
    B = np.zeros((self.T, M, N))
    
    U[0] = U0
    B[0] = B0
    
    for t in range(1, self.T + 1):
      #U[t] = U[t-1] + self.RHS(U[t-1], B[t-1], V, args) * self.dt
      #B[t] = B[t-1] + self.g(U[t-1], B[t-1]) * self.dt
      
      #U[t] = np.linalg.solve(self.RHS(U[t-1], B[t-1], V, args) * self.dt, U[t-1])
      #B[t] = B[t-1] + self.g(U[t-1], B[t-1]) * self.dt
      #B[t] = np.linalg.solve(self.g(U[t-1], B[t-1]) * self.dt, B[t-1])
      
      solU = optimize.root(self.FFF, U[t-1], args=(V, B[t-1], U[t-1],), method='lm')
      B[t] = B[t-1] + self.g(U[t-1], B[t-1]) * self.dt
      
      #print(solU)
      U[t] = solU.x.reshape(M, N)
      
      plt.imshow(U[t])
      plt.show()
      
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
  def solvePDE(self, spatial='fd', time='rk4', s1=0, s2=0):
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
      
      V1 = self.v[0](X, Y, 0)
      V2 = self.v[1](X, Y, 0)
      
      V = (V1, V2)
      
      args = (Dx[1:-1, 1:-1], Dy[1:-1, 1:-1], D2x[1:-1, 1:-1], D2y[1:-1, 1:-1])
      
    elif spatial == "fd":
      # Grid for functions evaluation
      X, Y = np.meshgrid(self.x, self.y)
      
      U0 = self.u0(X, Y) # Temperature initial condition
      B0 = self.beta0(X, Y) # Fuel initial condition
      B0[0,:] = np.zeros(self.N)
      B0[:,0] = np.zeros(self.M)
      B0[-1,:] = np.zeros(self.N)
      B0[:,-1] = np.zeros(self.M)
      V1 = self.v[0](X, Y, 0)
      V2 = self.v[1](X, Y, 0)
      V = (V1, V2) # Vector field
      
      Dx = FD1Matrix(self.N, self.dx, self.sparse)
      Dy = FD1Matrix(self.M, self.dy, self.sparse)
      D2x = FD2Matrix(self.N, self.dx, self.sparse)
      D2y = FD2Matrix(self.M, self.dy, self.sparse)

      args = (Dx[1:-1, 1:-1], Dy[1:-1, 1:-1], D2x[1:-1, 1:-1], D2y[1:-1, 1:-1])
    else:
      print("Spatial method error")
    
    # Time
    if time == 'rk4':
      U, B = self.solveRK4(U0, B0, V, args)
    elif time == 'euler': 
      U, B = self.solveEuler(U0, B0, V, args)
    elif time == 'last':
      U, B = self.solveRK4last(U0, B0, V, args)
    elif time == 'vec':
      U, B = self.solveRK4vec(U0, B0, V, args)
    elif time == 'veclast':
      U, B = self.solveRK4vecLast(U0, B0, V, args)
    elif time == 'eulveclast':
      U, B = self.solveEulerVecLast(U0, B0, V, args)
    elif time == 'odeint':
      U, B = self.solveODEIntLast(U0, B0, V, args)
    else:
      print("Time method error")
        
    return U, B
  
  def solvePDEData(self, spatial='fd', time='rk4'):
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
      B0[0,:] = np.zeros(self.N)
      B0[:,0] = np.zeros(self.M)
      B0[-1,:] = np.zeros(self.N)
      B0[:,-1] = np.zeros(self.M)
      #V1 = self.v[0](X, Y, 0)
      #V2 = self.v[1](X, Y, 0)
      #V = (V1, V2) # Vector field
      V = self.v
      
      Dx = FD1Matrix(self.N, self.dx, self.sparse)
      Dy = FD1Matrix(self.M, self.dy, self.sparse)
      D2x = FD2Matrix(self.N, self.dx, self.sparse)
      D2y = FD2Matrix(self.M, self.dy, self.sparse)

      args = (Dx[1:-1, 1:-1], Dy[1:-1, 1:-1], D2x[1:-1, 1:-1], D2y[1:-1, 1:-1])
    else:
      print("Spatial method error")
    
    # Time
    if time == 'rk4':
      U, B = self.solveRK4Data(U0, B0, V, args)
    elif time == 'euler': 
      U, B = self.solveEuler(U0, B0, V, args)
    else:
      print("Time method error")
        
    return U, B

    
    
    