import cupy as cp

# Generic Gaussian #
def G(x, y, s):
    return cp.exp(-(x ** 2 + y ** 2) / s)

# PDE FUNCTIONS #
def K(u, kap, eps):
    return kap * (1 + eps * u) ** 3 + 1

def Ku(u, kap, eps):
    return 3 * eps * kap * (1 + eps * u) ** 2

def f(u, b, eps, alp, s):
    return s(u) * b * cp.exp(u / (1 + eps * u)) - alp * u

def g(u, b, eps, q, s):
    return -s(u) * (eps / q) * b * cp.exp(u / (1 + eps * u))

def H(u, upc):
    S = cp.zeros_like(u)
    S[u >= upc] = 1.0
    return S

# Others #
def sigmoid(u, k=.5):
    return 1 / (1 + cp.exp(-k * scale(u))) #0.5 * (1 + cp.tanh(k * self.scale(u)))

def scale(u, a=-10, b=10):
    return (b - a) * (u - cp.min(u)) / (cp.max(u) - cp.min(u)) + a

# Fuel boundary
def b0_bc(B):
    rows, cols = B.shape
    B[ 0,:] = cp.zeros(cols)
    B[-1,:] = cp.zeros(cols)
    B[:, 0] = cp.zeros(rows)
    B[:,-1] = cp.zeros(rows)
    return B
    
def b0_af2002(x, y):
    cp.random.seed(777)
    br = lambda x, y: cp.round(cp.random.uniform(size=(x.shape)), decimals=2)
    return b0_bc(br(x, y))