import numpy as np
from .time import Euler, RK4

class Integration:

	def __init__(self, Nt, t_lim, method, last):
		self.Nt = Nt
		self.t_min, self.t_max = t_lim
		self.last = last

		# Time domain
		self.t = np.linspace(self.t_min, self.t_max, self.Nt)
		self.dt = self.t[1] - self.t[0]

		# Integration method
		self.method = method
		if self.method == 'Euler':
			self.time_integration = Euler
		elif self.method == 'RK4':
			self.time_integration = RK4
		else:
			raise Exception("Time integration method error. Please select available time approximation method.")

	def getT(self):
		return self.t

	def integration(self, t, F, y0):
		return self.time_integration(t, F, y0, self.last)


