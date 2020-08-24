"""Right hand side approximation using Finite Difference.
"""
import numpy as np
from .diffmat import FD1Matrix, FD2Matrix

class FiniteDifference:

    def __init__(self, Nx, Ny, x_lim, y_lim, order=2, sparse=False, cmp=(1, 1, 1), **kwargs):
        # Domain info
        self.Nx = Nx
        self.Ny = Ny
        self.x_min, self.x_max = x_lim # x \in [x_min, x_max]
        self.y_min, self.y_max = y_lim # y \in [y_min, y_max]
        self.x = np.linspace(self.x_min, self.x_max, self.Nx) # Create x array
        self.y = np.linspace(self.y_min, self.y_max, self.Ny) # Create y array
        self.X, self.Y = np.meshgrid(self.x, self.y) # X, Y meshgrid
        self.dx = self.x[1] - self.x[0] # \Delta x
        self.dy = self.y[1] - self.y[0] # \Delta y

        # Others params
        self.order = order # Finite difference order of accuracy (2 or 4)
        self.sparse = sparse # Sparse differentiation matrices
        self.cmp = cmp # Components of models
        
        # Physical Model Functions
        self.v = kwargs['v']
        self.f = kwargs['f']
        self.g = kwargs['g']
        self.kap = kwargs['kap']

        self.K = kwargs['K']
        self.Ku = kwargs['Ku']

        # Differentiation matrices
        self.Dx = FD1Matrix(self.Nx, self.dx, self.order, self.sparse)
        self.Dy = FD1Matrix(self.Ny, self.dy, self.order, self.sparse)
        self.D2x = FD2Matrix(self.Nx, self.dx, self.order, self.sparse)
        self.D2y = FD2Matrix(self.Ny, self.dy, self.order, self.sparse)

        # Vector field wrapper. Check if v is lambda or numpy array
        if type(self.v) is np.ndarray:
            self.V = lambda t: self.v[t]
        else:
            self.V = lambda t: self.v(self.X, self.Y, t)

    def getX(self):
        return self.x

    def getY(self):
        return self.y

    def getMesh(self):
        return self.X, self.Y
    
    def RHS(self, t, y):
        """
        Compute right hand side of PDE:

        .. math::
            \begin{split}
                u_{t} &= \Delta u - \mathbf{v} \cdot \nabla u + f(u, \beta) \\
                \beta_{t} &= g(u, \beta)
            \end{split}

        Parameters
        ----------
        t : array_like, shape (Nt + 1, )
            Time discrete variable.
        y : array_like, shape (2 * Ny * Nx) 
            Temperature and fuel variables vectorized.

        Returns
        -------
        y : array_like, shape (2 * Ny * Nx)
            New temperature and fuel variables vectorized.
            
        """
        # Vector field evaluation
        V1, V2 = self.V(t)

        
        # Recover u and b from y
        U = np.copy(y[:self.Ny * self.Nx].reshape((self.Ny, self.Nx), order='F'))
        B = np.copy(y[self.Ny * self.Nx:].reshape((self.Ny, self.Nx), order='F'))

        # Compute derivatives
        if self.sparse:
            Ux, Uy = (self.Dx.dot(U.T)).T, self.Dy.dot(U) # grad(U) = (u_x, u_y)
            Uxx, Uyy = (self.D2x.dot(U.T)).T, self.D2y.dot(U) # u_{xx} and u_{yy}
        else:
            Ux, Uy = np.dot(U, self.Dx.T), np.dot(self.Dy, U) # grad(U) = (u_x, u_y)
            Uxx, Uyy = np.dot(U, self.D2x.T), np.dot(self.D2y, U) # u_{xx} and u_{yy}
            
        # Laplacian of u
        lapU = Uxx + Uyy

        # Compute diffusion
        if self.K is not None and self.Ku is not None: # Using K(U) diffusion function
            K = self.K(U) 
            Kx = self.Ku(U) * Ux #(Dx.dot(K.T)).T
            Ky = self.Ku(U) * Uy #Dy.dot(K)
            diffusion = Kx * Ux + Ky * Uy + K * lapU
        else: # Diffusion is constant with value \kappa
            # \kappa \Delta u = \kappa (u_{xx} + u_{yy}) or \kappa lap(U)
            diffusion = self.kap * lapU 
        
        convection = Ux * V1 + Uy * V2 # v \cdot grad u.    
        reaction = self.f(U, B) # eval fuel

        # Include or not the model components
        diffusion *= self.cmp[0]
        convection *= self.cmp[1]
        reaction *= self.cmp[2]
        
        # Compute RHS
        Uf = diffusion -  convection + reaction 
        Bf = self.g(U, B)
        
        # Add boundary conditions
        Uf, Bf = self.boundaryConditions(Uf, Bf)
        
        # Build y = [vec(u), vec(\beta)]^T and return
        return np.r_[Uf.flatten('F'), Bf.flatten('F')] 

    def boundaryConditions(self, U, B):
        """Add Dirichlet boundary conditions (BC).
        Let \Gamma the domain boundary, then Dirichlet boundary condition is
        U_{\Gamma} = 0, B_{\Gamma} = 0

        Parameters
        -----------
        U: array_like, shape (Ny, Nx)
            Temperature approximation without BC.
        B: array_like, shape (Ny, Nx)
            Fuel approximation without  BC

        Returns
        -------
        Ub: array_like, shape (Ny, Nx)
            Temperature approximation with BC
        Bb: array_like, shape (Ny, Nx) 
            Fuel approximation with BC

        """
        Ub = np.copy(U)
        Bb = np.copy(B)

        # Only Dirichlet: 
        Ub[ 0,:] = np.zeros(self.Nx)
        Ub[-1,:] = np.zeros(self.Nx)
        Ub[:, 0] = np.zeros(self.Ny)
        Ub[:,-1] = np.zeros(self.Ny)
        
        Bb[0 ,:] = np.zeros(self.Nx)
        Bb[-1,:] = np.zeros(self.Nx)
        Bb[:, 0] = np.zeros(self.Ny)
        Bb[:,-1] = np.zeros(self.Ny)

        return Ub, Bb

    # def evalV(self, t):
    #     if type(self.v) is tuple:
    #         return self.v[0](self.X, self.Y, t), self.v[1](self.X, self.Y, t)
    #     else
    #         return self.v[t, 0], self.v[t, 1]

    def reshaper(self, y, Nt=None):
        """Reshape function to restore correct size.

        Parameters
        ----------
        y : array_like
            Approximation array.
        Nt : int, optional
            Number of time steps, by default None.

        Returns
        -------
        U: array_like
            Temperaturea approximation array.
        B : array_like
            Fuel approximation array.

        """
        if Nt is None: # Just last approximation
            U = y[:self.Ny * self.Nx].reshape(self.Ny, self.Nx, order='F')
            B = y[self.Ny * self.Nx:].reshape(self.Ny, self.Nx, order='F')
        else: # Reshape all approximations
            U = y[:, :self.Ny * self.Nx].reshape(Nt, self.Ny, self.Nx, order='F')
            B = y[:, self.Ny * self.Nx:].reshape(Nt, self.Ny, self.Nx, order='F')

        return U, B