"""
Differentiation matrices

@author: dsanmartin
"""
import numpy as np
import scipy as sp
from scipy.linalg import circulant, toeplitz
from scipy.sparse import csr_matrix

# First derivative with Finite Difference Matrix
def FD1Matrix(N, h, sparse=False):
  """
  Compute first derivative using Finite Difference Matrix
  with central difference, assuming periodic boundary conditions.
  
  Parameters
  ----------
  N : int
      Number of nodes.
  h : float
      Step size.
  sparse : boolean
      If true, return sparse matrix.
      
  Returns
  -------
  D1 : (N, N) ndarray
      Central difference finite difference matrix; or
  sD1 : sparse-like
      Central difference finite difference sparse matrix.
  """
  d1 = np.zeros(N)
  d1[1] = -1
  d1[-1] = 1
  
  D1 = (2*h) ** -1 * circulant(d1)
  
  if sparse:
    sD1 = csr_matrix(D1)
    return sD1
  else:
    return D1
  
# Second derivative with Finite Difference Matrix
def FD2Matrix(N, h, sparse=False):
  """
  Compute second derivative using Finite Difference Matrix
  with central difference, assuming periodic boundary conditions.
  
  Parameters
  ----------
  N : int
      Number of nodes.
  h : float
      Step size.
  sparse : boolean
      If true, return sparse matrix.
      
  Returns
  -------
  D2 : (N, N) ndarray
      Central difference finite difference matrix; or
  sD2 : sparse-like
      Central difference finite difference sparse matrix.
  """
  d2 = np.zeros(N)
  d2[0] = -2
  d2[1] = 1
  d2[-1] = 1
  
  D2 = h ** -2 * circulant(d2)
  
  if sparse:
    sD2 = csr_matrix(D2)
    return sD2
  else:
    return D2

# Chebyshev differentiation matrix
def chebyshevMatrix(N):
  """
  Compute derivative using Chebyshev differentiation Matrix.
  Python version of Trefethen (Spectral methods) implementation.
  
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

# First derivative with Interpolant of 4th degree
def P4D1Matrix(N, h, sparse=False):
  """
  Compute first derivative using interpolant of degree 4
  with central difference, assuming periodic boundary conditions.
  
  Parameters
  ----------
  N : int
      Number of nodes.
  h : float
      Step size.
  sparse : boolean
      If true, return sparse matrix.
      
  Returns
  -------
  D1 : (N, N) ndarray
      4th degree interpolant difference matrix; or
  sD1 : sparse-like
      4th degree interpolant difference sparse matrix.
  """
  d1 = np.zeros(N)
  d1[1] = -2/3
  d1[2] = 1/12
  d1[-2] = -1/12
  d1[-1] = 2/3
  
  D1 = h ** -1 * circulant(d1)
  
  if sparse:
    sD1 = sp.csr_matrix(D1)
    return sD1
  else:
    return D1

# First derivative with Interpolant of 4th degree
def SD1Matrix(N, h, sparse=False):
  """
  Compute first derivative using interpolant of degree 4
  with central difference, assuming periodic boundary conditions.
  
  Parameters
  ----------
  N : int
      Number of nodes.
  h : float
      Step size.
  sparse : boolean
      If true, return sparse matrix.
      
  Returns
  -------
  D1 : (N, N) ndarray
      4th degree interpolant difference matrix; or
  sD1 : sparse-like
      4th degree interpolant difference sparse matrix.
  """
  c = np.zeros(N)
  j = np.arange(1, N)
  c[1:] = 0.5 *((-1)**j)*(np.tan(j*h/2.)**(-1))
  r = np.zeros(N)
  r[0] = c[0]
  r[1:] = c[-1:0:-1]
  D = toeplitz(c, r=r)

  if sparse:
    sD = sp.csr_matrix(D)
    return sD
  else:
    return D

def spectralDFFT(v, nu=1):
    if not np.all(np.isreal(v)):
        raise ValueError('The input vector must be real')
    N=v.shape[0]
    K=np.fft.fftfreq(N)*N
    iK=(1j*K)**nu
    v_hat=np.fft.fft(v)
    w_hat=iK*v_hat
    if np.mod(nu,2)!=0:
        w_hat[int(N/2)]=0
    return np.real(np.fft.ifft(w_hat))

def spectralD2(N, h):
    #h=(2*np.pi/N)
    c=np.zeros(N)
    j=np.arange(1,N)
    c[0]=-np.pi**2/(3.*h**2)-1./6.
    #print(c)
    c[1:]=-0.5*((-1)**j)/(np.sin(j*h/2.)**2)
    #print(c)
    D2 = toeplitz(c)
    return D2