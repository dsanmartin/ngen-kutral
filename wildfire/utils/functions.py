"""Functions utilities.

Utils functions for implementation.

"""
import numpy as np

# Generic #
def G(x, y, s):
	"""Gaussian kernel.

	.. math::
		G(x, y) = \exp(-(x^2 + y^2) / s)

	Parameters
	----------
	x : float or array_like
			x value.
	y : float or array_like
			y value.
	s : float
			Gaussian shape parameter.

	Returns
	-------
	float or array_like
			Gaussian function
	"""
	return np.exp(-(x ** 2 + y ** 2) / s)

# PDE FUNCTIONS #
def K(u, kap, eps):
	"""Compute diffusion function 

	.. math::
		K(u) = \kappa \, (1 + \varepsilon u)^3 + 1

	Parameters
	----------
	u : array_like
			Temperature variable.
	kap : float
			Diffusion parameter.
	eps : float
			Inverse of activation energy.

	Returns
	-------
	array_like
			Evaluation of K function.
	"""
	return kap * (1 + eps * u) ** 3 + 1

def Ku(u, kap, eps):
	"""Derivative of K with respect to u.

	.. math:
		\dfrac{\partial K}{\partial u} = K_{u} = 3\,\varepsilon \kappa\, (1 + \varepsilon\, u)^2
	Parameters
	----------
	u : array_like
			Temperature variable.
	kap : float
			Diffusion parameter.
	eps : float
			Inverse of activation energy.

	Returns
	-------
	array_like
			Ku evaluation.
	"""
	return 3 * eps * kap * (1 + eps * u) ** 2

def f(u, b, eps, alp, s):
	"""Temperature-fuel reaction function.

	Parameters
	----------
	u : array_like
			Temperature value.
	b : array_like
			Fuel value.
	eps : float
			Inverse of activation energy parameter.
	alp : float
			Natural convection parameter.
	s : function or lambda
			Step function.

	Returns
	-------
	array_like
			Reaction function.
	"""
	return s(u) * b * np.exp(u / (1 + eps * u)) - alp * u

def g(u, b, eps, q, s):
	"""RHS of fuel PDE.

	Parameters
	----------
	u : array_like
			Temperature value
	b : array_like
			Fuel value.
	eps : float
			Inverse of activation energy parameter.
	q : float
			Reaction heat parameter.
	s : function or lambda
			Step function.

	Returns
	-------
	array_like
			Fuel RHS PDE.
	"""
	return -s(u) * (eps / q) * b * np.exp(u / (1 + eps * u))

def H(u, upc):
	"""2D heaviside funcion

	Parameters
	----------
	u : array_like
			Temperature value
	upc : float
			Phase change threshold.

	Returns
	-------
	array_like
			Heaviside function evaluation.
	"""
	S = np.zeros_like(u)
	S[u >= upc] = 1.0
	return S

def sigmoid(u, k=.5):
	"""Sigmoid function.

	Parameters
	----------
	u : array_like
			Temperature value.
	k : float, optional
			Slope constant factor, by default .5

	Returns
	-------
	array_like
			Sigmoid evaluation.
	"""	
	return 1 / (1 + np.exp(-k * scale(u))) #0.5 * (1 + np.tanh(k * self.scale(u)))

def scale(u, a=-10, b=10):
	"""Scale function.

	Parameters
	----------
	u : array_like
			Temperature value.
	a : int, optional
			Minimum value, by default -10
	b : int, optional
			Maximum value, by default 10

	Returns
	-------
	array_like
			Scaled value of u.
	"""
	return (b - a) * (u - np.min(u)) / (np.max(u) - np.min(u)) + a