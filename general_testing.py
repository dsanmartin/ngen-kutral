import numpy as np
import matplotlib.pyplot as plt
import scipy.sparse as sp
from diffmat import FD1Matrix, FD2Matrix, chebyshevMatrix, \
  P4D1Matrix, SD1Matrix, spectralDFFT, spectralD2
from scipy.interpolate import interp1d

#%%
f = lambda x: np.sin(x) # Function to compute the derivative
fd1 = lambda x: np.cos(x) # First derivative
fd2 = lambda x: -np.sin(x) # Second derivative


#f = lambda x: np.exp(np.sin(np.pi*x)) # Function to compute the derivative
#fd1 = lambda x: np.pi*np.exp(np.sin(np.pi*x)) * np.cos(np.pi*x)# # First derivative
#fd2 = lambda x: np.pi**2*(-np.exp(np.sin(np.pi*x)))* (np.sin(np.pi*x) - np.cos(np.pix)**2)# # Second derivative

# Linear transformation
xt = lambda t: (b-a)/2*t + (b+a)/2
tx = lambda x: 2/(b-a)*x-(b+a)/(b-a)
#%%
"""
Chebyshev for any domain
 
x \in [-a, b]
t \in [-1, 1]

x = (b-a)/2*t + (b+a)/2
t = 2/(b-a)*x-(b+a)/(b-a)

dt/dx = 2/(b-a)
dx/dt = (b-a)/2

f = f(x(t))

First derivative
df/dx = df/dt*dt/dx


Second derivative
d(df/dx)/dx = d(df/dt)/dx*dt/dx
 = d^2f/dt^2*(dt/dx)^2
"""

nodes = np.array([25, 100, 500, 1000])
nn = len(nodes)

# Error for first derivative
fd_error = np.zeros(nn)
cm_error = np.zeros(nn)
pm_error = np.zeros(nn)
sm_error = np.zeros(nn)
sf_error = np.zeros(nn)

# Error for second derivative
fd_2_error = np.zeros(nn)
cm_2_error = np.zeros(nn)
sm_2_error = np.zeros(nn)
sf_2_error = np.zeros(nn)

for i in range(nn):
  N = nodes[i]
  a, b = -np.pi, np.pi
  x = np.linspace(a, b, N, endpoint=False)
  dx = x[1] - x[0]
  
  # Eval function 
  ff = f(x)
  fx = fd1(x)
  fxx = fd2(x)
  
  # Compute differentiation matrices for first derivative
  D1 = FD1Matrix(N, dx)
  D1c, xc = chebyshevMatrix(N)
  D1p = P4D1Matrix(N, dx)
  D1s = SD1Matrix(N, dx)
  
  # Eval for Chebyshev
  ffc = f(xt(xc))
  ffcx = fd1(xt(xc))
  ffcxx = fd2(xt(xc))
  
  
  # First derivative
  f1n = np.dot(D1, ff)
  f1nc = np.dot(D1c, ffc)
  fi = interp1d(xc, f1nc, kind='cubic')
  fii = (2/(b-a)) * fi(tx(x))
  f1np = np.dot(D1p, ff)
  f1ns = np.dot(D1s, ff)
  f1nfft = spectralDFFT(ff)
  
  fd_error[i] = np.linalg.norm(f1n - fx, np.inf)
  pm_error[i] = np.linalg.norm(f1np - fx, np.inf)
  sm_error[i] = np.linalg.norm(f1ns - fx, np.inf)
  cm_error[i] = np.linalg.norm(fii - fx, np.inf)
  sf_error[i] = np.linalg.norm(f1nfft - fx, np.inf)
  
  # Compute differentiation matrices for second derivative
  D2 = FD2Matrix(N, dx)
  D2c = np.dot(D1c, D1c)
  D2s = spectralD2(N, dx)
  
  # Second Derivative
  f2n = np.dot(D2, ff)
  f2nc = np.dot(D2c, ffc)  
  f2i = interp1d(xc, f2nc, kind='cubic')
  f2ii = (2/(b-a)) ** 2 * f2i(tx(x))
  f2ns = np.dot(D2s, ff)
  f2nfft = spectralDFFT(ff, 2)
  
  fd_2_error[i] = np.linalg.norm(f2n - fxx, np.inf)
  cm_2_error[i] = np.linalg.norm(f2ii - fxx, np.inf)
  sm_2_error[i] = np.linalg.norm(f2ns - fxx, np.inf)
  sf_2_error[i] = np.linalg.norm(f2nfft - fxx, np.inf)
  


#print("FD Matrix error: ", np.linalg.norm(f1n - fx))
#print("P4 Matrix error: ", np.linalg.norm(f1np - fx))
#print("Sinc Matrix error: ", np.linalg.norm(f1ns - fx))
#print("Chebyshev Matrix error: ", np.linalg.norm(fii - fx))

#%%
#plt.plot(x, ff)
plt.plot(x, fx, label="Real")
plt.plot(x, f1n, '-o', label="FD")
plt.plot(x, f1np, linestyle='-.', label="P4")
plt.plot(x, f1ns, '-x', label="Sinc")
plt.plot(x, fii, linestyle='--', label="Chebyshev")
plt.plot(x, f1nfft, ':', label='FFT')
plt.title("First derivative")
plt.legend()
plt.show()

  
plt.plot(x, np.abs(f1n - fx), '-o', label="FD")
plt.plot(x, np.abs(f1np - fx), '-.', label="P4")
plt.plot(x, np.abs(f1ns - fx), '-x', label="Sinc")
plt.plot(x, np.abs(fii - fx), '--', label="Chebyshev")
plt.plot(x, np.abs(f1nfft - fx), ':', label="FFT")
plt.yscale('log')
plt.legend()
plt.grid(True)
plt.show()

plt.plot(nodes, fd_error, '-o', label="FD")
plt.plot(nodes, pm_error, '-s', label="P4")
plt.plot(nodes, sm_error, '-x', label="Sinc")
plt.plot(nodes, cm_error, '-d', label="Chebyshev")
plt.plot(nodes, sf_error, ':', label="FFT")
plt.yscale('log')
plt.legend()
plt.grid(True)
plt.show()

  
# Second derivative #
  
#plt.plot(x, fxx, label="Real")
#plt.plot(x, f2n, '-o', label="FD")
##plt.plot(x, f1np, linestyle='-.', label="P4")
#plt.plot(x, f2ns, '-x', label="Sinc")
#plt.plot(x, f2ii, linestyle='--', label="Chebyshev")
#plt.plot(x, f2nfft, ':', label='FFT')
#plt.title("Second derivative")
#plt.legend()
#plt.show()

#plt.plot(x, np.abs(f2n - fx), '-o', label="FD")
##plt.plot(x, np.abs(f2np - fx), '-.', label="P4")
#plt.plot(x, np.abs(f2ns - fx), '-x', label="Sinc")
#plt.plot(x, np.abs(f2ii - fx), '--', label="Chebyshev")
#plt.plot(x, np.abs(f2nfft - fx), ':', label="FFT")
#plt.yscale('log')
#plt.legend()
#plt.show()
#  
#plt.plot(nodes, fd_2_error, '-o', label="FD")
##plt.plot(nodes, pm_error, '-s', label="P4")
#plt.plot(nodes, sm_error, '-x', label="Sinc")
#plt.plot(nodes, cm_2_error, '-d', label="Chebyshev")
#plt.plot(nodes, sf_2_error, ':', label="FFT")
#plt.title('Inf norm error')
#plt.yscale('log')
#
#plt.legend()
#plt.grid(True)
#plt.show()

#%%

fff = lambda x, y: x**3 + y**3#np.sin(x + y)
fffx = lambda x, y: 3*x**2#np.cos(x + y)
fff2x = lambda x, y: 6*x#-np.sin(x + y)
fff2y = lambda x, y: 6*y#-np.sin(x + y)
fff_lap = lambda x, y: 6*x + 6*y #-2*np.sin(x + y)

X, Y = np.meshgrid(x, x)
F = fff(X, Y)

fff2xn = dx**-2 * np.dot(F, D2.T)

fffxn = 1/(2*dx) * np.dot(D1, F)

fff2xg = np.gradient(np.gradient(F, dx, edge_order=2, axis=1), dx, edge_order=2, axis=1)

ffflapn = dx**-2 * (np.dot(F, D2.T) + np.dot(D2, F))

#
#plt.contourf(X, Y, F)
#plt.show()
#plt.contourf(X, Y, fff2x(X, Y))
#plt.colorbar()
#plt.show()
plt.contourf(X, Y, fff_lap(X, Y))
plt.colorbar()
plt.show()
plt.contourf(X[1:-1,1:-1], Y[1:-1,1:-1], ffflapn[1:-1,1:-1])
plt.show()


#%%

N = 100
M = 100
R = 8.3143
x = np.linspace(-2, 2, N)
y = np.linspace(-2, 2, M)
dx = x[1] - x[0]
dy = y[1] - y[0]

X, Y = np.meshgrid(x, y)

#f = lambda x, y: x**3 + y**3 + x**2 + y**2 + x + y + 8*np.cos(x*y)
#EA = f(X, Y)
#A = 1
#
#r = lambda x, t: A*np.exp(-x/(R*t))
#T = lambda x: x + 1
#
#plt.plot(x, T(x))
#plt.plot(T(x), r(25, T(x)))
#
##plt.plot(x, r(0, x))
#plt.contourf(X, Y, EA)

f = lambda x, y: x*np.exp(-(x**2 + y**2)) + 1
f2 = lambda x, y: -(np.cos(x)**2 + np.cos(y)**2)**2 + 5

F = f(X, Y)
F2 = f2(X, Y)


dfdx = np.gradient(F, dx, axis=1)
dfdy = np.gradient(F, dy, axis=0)

c1 = plt.contourf(X, Y, F, cmap=plt.cm.jet)
plt.colorbar(c1)
plt.show()

c2 = plt.contourf(X, Y, F2, cmap=plt.cm.jet)
plt.colorbar(c2)
plt.show()
#s = 4
#plt.quiver(X[::s, ::s], Y[::s, ::s], dfdx[::s, ::s], dfdy[::s, ::s])

c3 = plt.contourf(X, Y, np.exp(F2/F), cmap=plt.cm.jet)
plt.colorbar(c3)

plt.show()
#%% GRADIENT TEST
import numpy as np
import matplotlib.pyplot as plt

M, N = 100, 100


x = np.linspace(-1, 1, N)
y = np.linspace(-1, 1, M)
dx = 2/N
dy = 2/M

f = lambda x, y: 1e1*(np.exp(-25*((x+.5)**2 + (y)**2)) + np.exp(-25*((x-.5)**2 + y**2)))
fx = lambda x, y: -500*((x+.5) * np.exp(-25*((x+.5)**2 + y**2)) + (x-.5) * np.exp(-25*((x-.5)**2 + y**2)))
fy = lambda x, y: -500*(y * np.exp(-25*((x+.5)**2 + y**2)) + y * np.exp(-25*((x-.5)**2 + y**2)))


X, Y = np.meshgrid(x, y)
F = f(X, Y)
Fx = np.gradient(F, dx, axis=1)
Fy = np.gradient(F, dy, axis=0)

plt.contourf(X, Y, f(X, Y))

#plt.quiver(X[::4,::4], Y[::4,::4], Fx[::4,::4], Fy[::4,::4])
plt.quiver(X[::4,::4], Y[::4,::4], fx(X[::4,::4], Y[::4,::4]), fy(X[::4,::4],Y[::4,::4]))

plt.show()
#%%
import numpy as np
import matplotlib.pyplot as plt

f = lambda x,y: 1e1*np.exp(-(1e1*(x)**2 + 1e3*(y-0.9)**2))

x = np.linspace(-1, 1, 100)
y = np.linspace(-1, 1, 100)

X, Y = np.meshgrid(x, y)

F = f(X, Y)

plt.contourf(X, Y, F)
plt.show()