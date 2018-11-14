import numpy as np
import scipy.special as spsp

def li(x, xk, i):
  return np.prod(np.delete((x - xk), i))

def weights(n, points="cheb"):
  j = np.arange(n) 
  if points == "cheb":
    nodes = (-1.0) ** j
    nodes[0] *= 0.5
    nodes[-1] *= 0.5
    return nodes
  else:
    comb = spsp.binom(n-1, j)
    return (-1.0) ** j * comb
    

def BL1Done(x, xi, f, wi):
  if x in xi: # special case
    return f[np.where(x == xi)]
  else:
    xdiff = x - xi
    #xdiff[np.where(xdiff == 0)] = 1
    den = np.dot(wi, 1/xdiff) # np.dot is faster than np.sum(wi/xdiff)
    num = np.dot(wi/xdiff, f)
    return num / den
  
def BL1D(x, xi, f, wi):
  #wi = np.array([1 / li(xi[i], xi, i) for i in range(len(xi))])
  xdiff = x.reshape(-1, 1) - xi.reshape(1, -1)
  xdiff[xdiff == 0] = 1 # For x = xi
  interp = np.dot(wi/xdiff, f) / np.dot(1/xdiff, wi)
  # Replace values where x = xi with real data
  interp[np.where(np.in1d(x, xi))] = f[np.where(np.in1d(xi, x))]
  return interp

def BL2Done(x, y, xi, yj, f):  
  if x in xi and y in yj:
    return f[np.where(y == yj), np.where(x == xi)]
  elif x in xi:
    return BL1Done(y, yj, f[:, np.where(x == xi)].flatten())
  elif y in yj:
    return BL1Done(x, xi, f[np.where(y == yj), :].flatten())
  else:
    wi = np.array([1 / li(xi[i], xi, i) for i in range(len(xi))])
    #wj = np.array([1 / li(yj[j], yj, j) for j in range(len(yj))])
    
    xdiff = x - xi
    #di = np.where(xdiff == 0)
    #ydiff = y - yj
    #xdiff[np.where(xdiff == 0)] = 1
    #ydiff[np.where(ydiff == 0)] = 1
    deni = np.dot(wi, 1/xdiff)
    #deni = np.dot(np.delete(wi, di), 1/np.delete(xdiff, di))
    #denj = np.dot(wj, 1/ydiff)
    bl1ds = np.array([BL1Done(y, yj, f[:,k].flatten()) for k in range(len(xi))])
    numi = np.dot(wi/xdiff, bl1ds)
    
    #num = sum(wi[k] / (x - xi[k]) * BL1D(y, yj, f[:,k].flatten()) for k in range(len(xi)) if x !=  xi[k])
    #den = sum(wi[k] / (x - xi[k]) for k in range(len(xi)) if x !=  xi[k] )
    return numi / deni

def BL2Dnp(x, y, xi, yj, f, wi, wj):  
  if x in xi and y in yj:
    return f[np.where(y == yj), np.where(x == xi)]
  elif x in xi:
    return BL1Done(y, yj, f[:, np.where(x == xi)].flatten(), wi)
  elif y in yj:
    return BL1Done(x, xi, f[np.where(y == yj), :].flatten(), wj)
  else:   
    xdiff = x - xi
    ydiff = y - yj
    deni = np.dot(wi, 1/xdiff)
    denj = np.dot(wj, 1/ydiff)
    yi = np.dot(f.T, wj/ydiff)
    num = np.dot(wi/xdiff, yi)
    den = deni * denj
    return num / den
  
def BL2Dsum(x, y, xi, yj, f, wi, wj):  
  if x in xi and y in yj:
    return f[np.where(y == yj), np.where(x == xi)]
  elif x in xi:
    return BL1Done(y, yj, f[:, np.where(x == xi)].flatten(), wi)
  elif y in yj:
    return BL1Done(x, xi, f[np.where(y == yj), :].flatten(), wi)
  else:
    #wi = np.array([1 / li(xi[i], xi, i) for i in range(len(xi))])
    
    num = sum(wi[k] / (x - xi[k]) * BL1Done(y, yj, f[:,k].flatten(), wi) for k in range(len(xi)))
    den = sum(wi[k] / (x - xi[k]) for k in range(len(xi)))
    return num / den

def interpolation2D(x, y, xi, yj, f, wi, wj, method):
  F = np.zeros((len(y), len(x)))
#  wi = weights(len(xi), points="cheb")
#  wj = weights(len(yj), points="cheb")
  #wi = np.array([1 / li(xi[i], xi, i) for i in range(len(xi))])
  #wj = np.array([1 / li(yj[j], yj, j) for j in range(len(yj))])
  
  for j in range(len(y)):
    for i in range(len(x)):
      F[j, i] = method(x[i], y[j], xi, yj, f, wi, wj)
      
  return F  

