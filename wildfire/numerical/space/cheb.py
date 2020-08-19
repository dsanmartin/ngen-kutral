import numpy as np
from scipy.interpolate import interp2d

class Chebyshev:

	def __init__():
		self.Dx, self.x = chebyshevMatrix(self.Nx-1)
		self.Dy, self.y = chebyshevMatrix(self.Ny-1)
		self.D2x = np.dot(self.Dx, self.Dx)
		self.D2y = np.dot(self.Dy, self.Dy)
		
		self.X, self.Y = np.meshgrid(self.x, self.y)
		
		self.X0 = 2 / (self.x_max - self.x_min)
		self.Y0 = 2 / (self.y_max - self.y_min)
		
		self.u0 = lambda x, y: self.u0(self.tx(x), self.tx(y))
		self.b0 = lambda x, y: self.b0(self.tx(x), self.tx(y))

	# Chebyshev approximation in space
	def RHS(self, t, y):
		"""
		Compute right hand side of PDE
		"""
		# Vector field evaluation
		V1 = self.v[0](self.X, self.Y, t)
		V2 = self.v[1](self.X, self.Y, t)
		
		Uf = np.copy(y[:self.Ny * self.Nx].reshape((self.Ny, self.Nx)))
		Bf = np.copy(y[self.Ny * self.Nx:].reshape((self.Ny, self.Nx)))
		U = Uf
		B = Bf

		# Compute derivatives
		if self.sparse:
			Ux, Uy = (self.Dx.dot(U.T)).T, self.Dy.dot(U) # grad(U) = (u_x, u_y)
			Uxx, Uyy = (self.D2x.dot(U.T)).T, self.D2y.dot(U) # 
		else:
			Ux, Uy = np.dot(U, self.Dx.T), np.dot(self.Dy, U)
			Uxx, Uyy = np.dot(U, self.D2x.T), np.dot(self.D2y, U)
			
		lapU = self.X0 ** 2 * Uxx + self.Y0 ** 2 * Uyy
		
		if self.complete:
			K = self.K(U) 
			Kx = self.Ku(U) * Ux #(Dx.dot(K.T)).T
			Ky = self.Ku(U) * Uy #Dy.dot(K)
			diffusion = Kx * Ux + Ky * Uy + K * lapU
		else:
			diffusion = self.kap * lapU # k \nabla U
		
		convection = self.X0 * Ux * V1 + self.Y0 * Uy * V2 # v \cdot grad u.    
		reaction = self.f(U, B) # eval fuel

		# Components
		diffusion *= self.cmp[0]
		convection *= self.cmp[1]
		reaction *= self.cmp[2]
		
		Uf = diffusion -  convection + reaction 
		Bf = self.g(U, B)
		
		# Boundary conditions
		Uf, Bf = self.boundaryConditions(Uf, Bf)
		
		return np.r_[Uf.flatten(), Bf.flatten()]

	# Domain transformations
	def tx(self, t):  
		return (self.x_max - self.x_min) * t / 2 + (self.x_max + self.x_min)/2 # [-1,1] to [xa, xb]
	
	def xt(self, x):
		return 2 * x / (self.x_max - self.x_min) - (self.x_max +self.x_min) / (self.x_max - self.x_min) # [xa, xb] to [-1, 1]