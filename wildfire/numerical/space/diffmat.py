"""
Space derivatives approximation using differentiation matrices

Implements differentiation matrices for:
	- Finite Differences of second and fourth order of accuracy 
	- Chebyshev differentiation matrix

Details in: Trefethen, L. N. (2000). Spectral methods in MATLAB. Society for industrial and applied mathematics.
https://doi.org/10.1137/1.9780898719598
"""
import numpy as np
import scipy as sp
from scipy.linalg import circulant, toeplitz
from scipy.sparse import csr_matrix

# First derivative with Finite Difference Matrix
def FD1Matrix(N, h, acc=2, sparse=False):
	"""
	Compute first derivative using Finite Difference Matrix
	with O(h^acc) of accuracy.
	
	Parameters
	----------
	N 		 : int
					Number of nodes.
	h 		 : float
					Step size.
	acc		 : int
					Order of accuracy (2 or 4)
	sparse : boolean
					If true, return sparse matrix.
			
	Returns
	-------
	D1 : (N, N) ndarray
			Finite difference dense matrix; or
	sD1 : sparse-like
			Finite difference sparse matrix.
	"""
	d1 = np.zeros(N)

	if acc == 2: # Coefficients for second order
		d1[1] = -1/2
		d1[-1] = 1/2
	elif acc == 4: # Coefficients for fourth order
		d1[1] = -2/3
		d1[2] = 1/12
		d1[-1] = 2/3
		d1[-2] = -1/12
	
	D1 = circulant(d1) # Central difference inside the domain
	
	# Finite difference at boundary. 
	# To keep accuracy, it uses forward and backward coefficients of second or fourth order
	if acc == 2:
		D1[0,:3] = np.array([-3/2, 2, -1/2]) # Forward difference at left boundary
		D1[-1,-3:] = np.array([1/2, -2, 3/2]) # Backward difference at right boundary
	elif acc == 4:
		D1[0,:5] = np.array([-25/12	, 4, -3, 4/3, -1/4]) # Forward difference at left boundary
		D1[-1,-5:] = np.array([1/4, -4/3, 3, -4, 25/12]) # Backward difference at right boundary

	D1 = D1 / h # Include step
	
	if sparse: 
		D1 = csr_matrix(D1)

	return D1
	
# Second derivative with Finite Difference Matrix
def FD2Matrix(N, h, acc=2, sparse=False):
	"""
	Compute second derivative using Finite Difference Matrix
	with O(h^acc) of accuracy.
	
	Parameters
	----------
	N 		 : int
					Number of nodes.
	h 		 : float
					Step size.
	acc 	 : int
					Order of accuracy (2 or 4)
	sparse : boolean
					If true, return sparse matrix.
			
	Returns
	-------
	D2 : (N, N) ndarray
			Finite difference dense matrix; or
	sD2 : sparse-like
			Finite difference sparse matrix.
	"""
	d2 = np.zeros(N)
	
	if acc == 2:
		d2[0] = -2
		d2[1] = 1
		d2[-1] = 1
	elif acc == 4:
		d2[0] = -5/2
		d2[1] = 4/3
		d2[2] = -1/12
		d2[-1] = 4/3
		d2[-2] = -1/12

	D2 = circulant(d2) 
	
	if acc == 2:
		D2[0,:4] = np.array([2, -5, 4, -1]) # Forward difference at left boundary
		D2[-1,-4:] = np.array([-1, 4, -5, 2]) # Backward difference at right boundary
	elif acc == 4:
		D2[0,:6] = np.array([15/4, -77/6, 107/6, -13, 61/12, -5/6]) # Forward difference at left boundary
		D2[-1,:6] = np.array([-5/6, 61/12, -13, 107/6, -77/6, 15/4]) # Backward difference at right boundary
	
	D2 = D2 / (h ** 2)
	
	if sparse: D2 = csr_matrix(D2)
	
	return D2

# Chebyshev differentiation matrix
def chebyshevMatrix(N):
	"""
	Compute derivative using Chebyshev differentiation Matrix.
	
	Parameters
	----------
	N : int
			Number of nodes.
			
	Returns
	-------
	D2 : (N+1, N+1) ndarray
			Chebyshev differentation matrix
	x : (N+1) array
			Chebyshev x domain
	"""
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