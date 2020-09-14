"""Paramters handler.
Wildfire has a lot of parameters, and this file tries to manipulate for non-dimensional transformations.

"""
import numpy as np

# Constants
R = 8.31446261815324 # Universal gas constant [J K^{-1} mol^{-1}] 
SIGMA = 5.670374419E-8 # Stefan-Boltzmann constant [W m^2 K^{-4}]

class Parameters:

    def __init__(self, k, delta, rho, C, h, E_A, H, A, U_inf, B_0):
        self.k = k
        self.delta = delta
        self.rho = rho
        self.C = C
        self.h = h
        self.E_A = E_A
        self.H = H
        self.A = A
        self.U_inf = U_inf
        self.B_0 = B_0

    def getKappa(self):
        return 4 * SIGMA * self.delta * self.U_inf ** 3 / self.k

    def getEpsilon(self):
        return R * self.U_inf / self.E_A

    def getQ(self):
        return self.H * self.B_0 / self.C / self.U_inf

    def getAlpha(self):
        return self.getT0() * self.h / self.rho / self.C

    def getUpc(self, Upc):
        return (Upc - self.U_inf) / self.getEpsilon() / self.U_inf

    def getT0(self):
        return self.getEpsilon() * np.exp(1 / self.getEpsilon()) / self.getQ() / self.A

    def getL0(self):
        return np.sqrt(self.getT0() * self.k / self.rho / self.C)
        
    def getTemperature(self, u):
        return self.U_inf * (self.epsilon * u + 1)

    def getFuel(self, b):
        return self.B_0 * b

    def getVector(self, v):
        return self.L0() * v / self.getT0()