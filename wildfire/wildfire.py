"""Numerical implementation of mathematical model for wildfire spreading.

Wildfire class to handle the numerical implementation.

"""
import numpy as np
from .numerical.space import FiniteDifference, FFTDerivatives
from .numerical.time import Integration
from .utils.functions import K, Ku, f, g, H, sigmoid

class Fire:
	
	def __init__(self, kap, eps, upc, alp, q, x_lim, y_lim, t_lim, **kwargs):
		"""Wildfire constructor.

		Parameters
		----------
		kap : float
				Diffusion parameter :math:`\kappa`.
		eps : float
				Inverse of activation energy :math:`\varepsilon`.
		upc : float
				Phase change threshold paramenter :math:`u_{pc}'.
		alp : float
				Natural convection parameter :math:`\alpha'.
		q : float
				Reaction heat parameter :math:`q`
		x_lim : tuple
				:math:`x` domain limits.
		y_lim : tuple
				:math:`y` domain limits.
		t_lim : tuple
				:math:`t` domain limits.
		**kwargs : dict
				Extra parameters.

		Other Parameters
		----------------
		complete : boolean
				Use complete model with diffusion function :math:`K(u)`.
		cmp : tuple
				Model components included. Tuple needs 3 boolean, (diffusion, convection, reaction).
		sf	: string
				Solid-gas phase function.
		"""
		
		# Physical Model parameters
		self.kap = kap
		self.eps = eps
		self.upc = upc
		self.alp = alp
		self.q = q
		
		# Domain limits
		self.x_min, self.x_max = x_lim
		self.y_min, self.y_max = y_lim
		self.t_min, self.t_max = t_lim
		
		# Others parameters
		self.complete = kwargs.get('complete', False)
		self.cmp = kwargs.get('components', (True, True, True))
		self.sf = kwargs.get('sf', 'step')

		# Define PDE functions #
		s = lambda u: H(u, self.upc) if self.sf == 'step' else sigmoid(u)
		self.f = lambda u, b: f(u, b, self.eps, self.alp, s)
		self.g = lambda u, b: g(u, b, self.eps, self.q, s)
		self.K = lambda u: K(u, self.kap, self.eps)
		self.Ku = lambda u: Ku(u, self.kap, self.eps)

	def solvePDE(self, Nx, Ny, Nt, u0, b0, v, space_method='fd', time_method='RK4', last=True, **kwargs):
		"""Solve numerical PDE.

		Parameters
		----------
		Nx : int
				Number of x nodes.
		Ny : int
				Number of y nodes.
		Nt : int
				Number of t nodes.
		u0 : lambda or array_like, shape (Ny, Nx)
				Temperature initial condition.
		b0 : lambda or array_like, shape (Ny, Nx)
				Fuel initial condition.
		v : lambda or array_like, shape (Nt, Ny, Nx)
				Wind vector field.
		space_method : str, optional
				Numerical method for space approximation. 
				'fd' for Finite Difference, 'fft' for FFT based, by default 'fd'.
		time_method : str, optional
				Numerical method for time approximation. 
				'Euler' for Euler method, 'RK4' for Runge-Kutta of fourth order, by default 'RK4'.
		last : boolean, optional
				Keep and return last approximation, by default True.
		**kwargs : dict
				Extra parameters.

		Returns
		-------
		t : array_like, shape (Nt + 1,)
				Time domain.
		X : array_like, shape (Ny, Nx)
				Space domain, x variable (meshgrid).
		Y : array_like, shape (Ny, Nx)
				Space domain, y variable (meshgrid).
		U	: array_like, shape (Nt + 1, Ny, Nx) or (Ny, Nx) if last is True.
				Temperature approximation.
		B : array_like, shape (Nt + 1, Ny, Nx) or (Ny, Nx) if last is True.

		Other Parameters
		----------------
		acc : int
				Accuracy for Finite Difference, by default 2.
		sparse : bool, 
				Sparse matrices for Finite Difference, by default False.

		Raises
		------
		Exception
				Error for invalid numerical method selection.

		Notes
		-----
		Numerical approximation of PDE by Method of Lines (MOL) [1].
		This method requires a RHS approximation, performed by finite difference 'fd' [2]  or  fast fourier transform 'fft' [3], 
		and then solve an IVP using 'Euler' or 'RK4' method. 

		Variables used:
		.. Temperature :math:`u(x,y,t)` is represented by `U`.
		.. Fuel :math:`\beta(x,y,t)` is representeted by `B`.
		.. Vector field with wind and topography :math:`\mathbf{v}(x,y,t)` is represented by `V`.
		
		The implementation needs to vectorize :math:`u` and :math:`b` into :math:`\mathbf{y}`, 
		and then apply the time solver using the RHS function.

		References
		----------
		.. [1] Sarmin, E. N., & Chudov, L. A. (1963). "On the stability of the numerical integration of systems of ordinary differential equations arising in the use of the straight line method". 
				USSR Computational Mathematics and Mathematical Physics, 3(6), 1537–1543. https://doi.org/10.1016/0041-5553(63)90256-8
		.. [2] San Martín, D., & Torres, C. E. (2018). "Ngen-Kütral: Toward an Open Source Framework for Chilean Wildfire Spreading". 
				In 2018 37th International Conference of the Chilean Computer Science Society (SCCC) (pp. 1–8). https://doi.org/10.1109/SCCC.2018.8705159
		.. [3] San Martín, D., & Torres, C. E. (2019). "Exploring a Spectral Numerical Algorithm for Solving a Wildfire Mathematical Model". 
				In 2019 38th International Conference of the Chilean Computer Science Society (SCCC) (pp. 1–7). https://doi.org/10.1109/SCCC49216.2019.8966412
		"""

		# Space approximation #
		if space_method == 'fd': # Finite Differences
			# Get finite difference extra parameters
			acc = kwargs.get('acc', 2)
			sparse = kwargs.get('sparse', False)

			# Create FD
			FD = FiniteDifference(Nx, Ny, (self.x_min, self.x_max), (self.y_min, self.y_max), 
				order=acc, sparse=sparse, cmp=self.cmp, v=v, K=None, f=self.f, g=self.g, kap=self.kap)

			# FD Mesh
			X, Y = FD.getMesh()

			# Get RHS using FD
			RHS = FD.RHS
			
			# Get reshaper for approximations
			reshaper = FD.reshaper

			# Initial condition evaluation
			U0 = u0 if type(u0) is np.ndarray else u0(X, Y)
			B0 = b0 if type(b0) is np.ndarray else b0(X, Y)

		elif space_method == 'fft':

			Nx -= 1 # Remove one node for periodic boundary
			Ny -= 1 # Remove one node for periodic boundary
			
			FFTD = FFTDerivatives(Nx, Ny, (self.x_min, self.x_max), (self.y_min, self.y_max), cmp=self.cmp,
				v=v, K=None, f=self.f, g=self.g, kap=self.kap)

			# FFT Mesh
			X, Y = FFTD.getMesh()

			# RHS using FFT
			RHS = FFTD.RHS

			# Reshaper for approximations
			reshaper = FFTD.reshaper

			# Remove last row and column if FFT is used (for boundary).
			if type(u0) is np.ndarray:
				U0 = u0[:-1, :-1]
			else:
				U0 = u0(X, Y)
				
			if type(b0) is np.ndarray:
				B0 = b0[:-1, :-1]
			else:
				B0 = b0(X, Y)

		else:
			raise Exception("Spatial method error. Please select available space approximation method.")

		# Time approximation
		Nt += 1 # Include initial condition
		integrator = Integration(Nt, (self.t_min, self.t_max), time_method, last)
		t = integrator.getT()

		# Vectorize variables for Method of Lines. [vec(U), vec(B)]^T
		y0 = np.zeros((2 * Ny * Nx))
		y0[:Ny * Nx] = U0.flatten('F')
		y0[Ny * Nx:] = B0.flatten('F')
		
		# Integration
		y = integrator.integration(t, RHS, y0)

		U, B = reshaper(y) if last else reshaper(y, Nt)

		if space_method == 'fft':
			X, Y = FFTD.getMesh(True) # Append boundary removed
			
		return t, X, Y, U, B 
