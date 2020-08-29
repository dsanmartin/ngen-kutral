"""Time approximation using numerical integration methods.

Solve Initial Value Problem (IVP) or ODE:
    .. math::
        \frac{dy}{dt} = F(t, y) \\
        y(t_0) = y_0

"""
import numpy as np
from .time import Euler, RK4, IVP

class Integration:

    def __init__(self, Nt, t_lim, method, last, vdata):
        """Integrator constructor.

        Parameters
        ----------
        Nt : int
            Number of time nodes.
        t_lim : tuple
            Boundary of time variable.
        method : str
            Method used to integrate.
        last : bool
            Return and keep only last approximation.
        vdata : bool
            If vector field is np.ndarray.

        Raises
        ------
        Exception
            Error if method is not implemented.
        """
        self.Nt = Nt
        self.t_min, self.t_max = t_lim
        self.last = last
        self.vdata = vdata

        # If vector field is data, index of numerical method is used as time variable.
        if self.vdata:
            self.t = np.arange(Nt)
        else:
            self.t = np.linspace(self.t_min, self.t_max, self.Nt)

        # Integration method
        self.method = method
        if self.method == 'Euler':
            self.time_integration = Euler
        elif self.method == 'RK4':
            self.time_integration = RK4
        elif self.method in ['RK45', 'RK23', 'DOP853', 'Radau', 'BDF', 'LSODA'] and not self.vdata:
            self.time_integration = lambda t, F, y0, last, vdata: IVP(t, F, y0, last, self.method)
        else:
            raise Exception("Time integration method error. Please select available time approximation method.")

    def getTime(self):
        return np.linspace(self.t_min, self.t_max, self.Nt)

    def solve(self, t, F, y0):
        return self.time_integration(t, F, y0, self.last, self.vdata)


