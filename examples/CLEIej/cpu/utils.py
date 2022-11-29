import numpy as np

# Generic Gaussian #
def G(x, y, s):
    return np.exp(-(x ** 2 + y ** 2) / s)

# PDE FUNCTIONS #
def K(u, kap, eps):
    return kap * (1 + eps * u) ** 3 + 1

def Ku(u, kap, eps):
    return 3 * eps * kap * (1 + eps * u) ** 2

def f(u, b, eps, alp, s):
    return s(u) * b * np.exp(u / (1 + eps * u)) - alp * u

def g(u, b, eps, q, s):
    return -s(u) * (eps / q) * b * np.exp(u / (1 + eps * u))

def H(u, upc):
    S = np.zeros_like(u)
    S[u >= upc] = 1.0
    return S

# Others #
def sigmoid(u, k=.5):
    return 1 / (1 + np.exp(-k * scale(u))) #0.5 * (1 + np.tanh(k * self.scale(u)))

def scale(u, a=-10, b=10):
    return (b - a) * (u - np.min(u)) / (np.max(u) - np.min(u)) + a

# Fuel boundary
def b0_bc(B):
    rows, cols = B.shape
    B[ 0,:] = np.zeros(cols)
    B[-1,:] = np.zeros(cols)
    B[:, 0] = np.zeros(rows)
    B[:,-1] = np.zeros(rows)
    return B
    
def b0_af2002(x, y):
    np.random.seed(777)
    br = lambda x, y: np.round(np.random.uniform(size=(x.shape)), decimals=2)
    return b0_bc(br(x, y))