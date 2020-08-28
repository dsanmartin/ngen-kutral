import numpy as np

MAX_FLOAT = 1e10 # Defined to check divergence.... 

class FFTDerivatives:

    def __init__(self, Nx, Ny, x_lim, y_lim, cmp=(1,1,1), **kwargs):
        # Physical domain
        self.Nx = Nx
        self.Ny = Ny
        self.x_min, self.x_max = x_lim # x \in [x_min, x_max]
        self.y_min, self.y_max = y_lim # y \in [y_min, y_max]
        # Spatial domain. Remove last node for periodic boundary assumption
        self.x = np.linspace(self.x_min, self.x_max, self.Nx, endpoint=False) 
        self.y = np.linspace(self.y_min, self.y_max, self.Ny, endpoint=False)
        self.dx = self.x[1] - self.x[0] # \Delta x
        self.dy = self.y[1] - self.y[0] # \Delta y
        self.X, self.Y = np.meshgrid(self.x, self.y) # X, Y mesh
        
        # Fourier domain
        eta = np.fft.fftfreq(self.Ny, d=self.dy/(2*np.pi)) 
        xi =  np.fft.fftfreq(self.Nx, d=self.dx/(2*np.pi))
        self.XI, self.ETA = np.meshgrid(xi, eta)

        # Physical model functions
        self.v = kwargs['v']
        self.f = kwargs['f']
        self.g = kwargs['g']
        self.kap = kwargs['kap']

        # Diffusion functions
        self.K = kwargs['K']
        self.Ku = kwargs['Ku']

        # Others params
        self.cmp = cmp # Components of models

        # Vector field wrapper. Check if v is lambda or numpy array
        if type(self.v) is np.ndarray:
            self.V = lambda t: self.v[t]
        else:
            self.V = lambda t: self.v(self.X, self.Y, t)
        
    def getMesh(self, full=False):
        if full:
            # Append boundary to space domain
            x = np.append(self.x, self.x_max)
            y = np.append(self.y, self.y_max)
            return np.meshgrid(x, y)
        else:
            return self.X, self.Y

    # FFT to approximate space derivatives
    def RHS(self, t, y):
        """Compute right hand side of PDE using Fast Fourier Transform.

        Parameters
        ----------
        t : array_like
            Time variable.
        y : array_like
            [description]

        Returns
        -------
        array_like
            [description]

        Raises
        ------
        Exception
            [description]
        """
        # Vector field evaluation
        V1, V2 = self.V(t)
        
        Uf = np.copy(y[:self.Ny * self.Nx].reshape((self.Ny, self.Nx), order='F'))
        Bf = np.copy(y[self.Ny * self.Nx:].reshape((self.Ny, self.Nx), order='F'))
        U = Uf
        B = Bf

        # Fourier transform
        Uhat = np.fft.fft2(U)
        
        # First derivative approximation
        Uhatx = 1j * self.XI * Uhat
        Uhaty = 1j * self.ETA * Uhat
        if self.Nx % 2 == 0: Uhatx[:, self.Nx // 2] = np.zeros(self.Ny)
        if self.Ny % 2 == 0: Uhaty[self.Ny // 2, :] = np.zeros(self.Nx)
        Ux = np.real(np.fft.ifft2(Uhatx))
        Uy = np.real(np.fft.ifft2(Uhaty))
        
        # Laplace operator
        lap = np.real(np.fft.ifft2((-(self.XI ** 2 + self.ETA ** 2)) * Uhat)) 

        # Compute diffusion
        if self.K is not None and self.Ku is not None: # Using K(U) diffusion function
            K = self.K(U) 
            Ku = self.Ku(U)
            #Kx = self.Ku(U) * Ux 
            #Ky = self.Ku(U) * Uy 
            #diffusion = Kx * Ux + Ky * Uy + K * lap
            diffusion = Ku * (Ux ** 2 + Uy ** 2) + K * lap
        else: # Diffusion is constant with value \kappa
            # \kappa \Delta u = \kappa (u_{xx} + u_{yy}) or \kappa lap(U)
            diffusion = self.kap * lap

        #diffusion = self.kap * lap # k \nabla U
        
        convection = Ux * V1 + Uy * V2 # v \cdot grad u.    
        reaction = self.f(U, B) # eval fuel
        
        # Components
        diffusion *= self.cmp[0]
        convection *= self.cmp[1]
        reaction *= self.cmp[2]

        # Compute RHS approximation
        Uf = diffusion - convection + reaction
        Bf = self.g(U, B)

        # Check if approximation diverges
        if np.any(np.isnan(Uf)) or np.any(np.isinf(Uf)) or np.any(np.isnan(Bf)) or  \
            np.any(np.isinf(Bf)) or np.any(Uf > MAX_FLOAT) or np.any(Bf > MAX_FLOAT):
            raise Exception("Numerical approximation diverges. Please check number of nodes in space or time.") 
        
        return np.r_[Uf.flatten('F'), Bf.flatten('F')]

    def reshaper(self, y, Nt=None):

        if Nt is None:
            # Get u and b from y
            U = y[:self.Ny * self.Nx].reshape(self.Ny, self.Nx, order='F')
            B = y[self.Ny * self.Nx:].reshape(self.Ny, self.Nx, order='F')

            # Stack first row/column to the last row/column (periodic boundary)
            U = np.concatenate((U, U[0].reshape(1,-1)), axis=0)
            U = np.concatenate((U, U[:,0].reshape(-1,1)), axis=1)
            B = np.concatenate((B, B[0].reshape(1,-1)), axis=0)
            B = np.concatenate((B, B[:,0].reshape(-1,1)), axis=1)
        else:
            # Get u and b from y
            U = y[:, :self.Ny * self.Nx].reshape(Nt, self.Ny, self.Nx, order='F')
            B = y[:, self.Ny * self.Nx:].reshape(Nt, self.Ny, self.Nx, order='F')

            U_bc = np.zeros((Nt, self.Ny + 1, self.Nx + 1))
            B_bc = np.zeros((Nt, self.Ny + 1, self.Nx + 1))
            U_bc[:,:-1,:-1] = U
            B_bc[:,:-1,:-1] = B

            for k in range(Nt):
                U_bc[k,-1,:-1] = U[k,0]
                U_bc[k,:,-1] = U_bc[k,:,0]
                B_bc[k,-1,:-1] = B[k,0]
                B_bc[k,:,-1] = B_bc[k,:,0]
    #          U_bc[k] = np.concatenate((U_bc[k], U[k,0].reshape(1,-1)), axis=0)
    #          U_bc[k] = np.concatenate((U_bc[k], U[k,:,0].reshape(-1,1)), axis=1)
    #          B_bc[k] = np.concatenate((B_bc[k], B[k,0].reshape(1,-1)), axis=0)
    #          B_bc[k] = np.concatenate((B_bc[k], B[k,:,0].reshape(-1,1)), axis=1)
            U = np.copy(U_bc)
            B = np.copy(B_bc)

        return U, B

