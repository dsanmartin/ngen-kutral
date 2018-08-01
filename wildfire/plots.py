import numpy as np
#import matplotlib
#matplotlib.use('agg')
import matplotlib.pyplot as plt
import matplotlib.colors as colors


def truncate_colormap(cmap, minval=0.0, maxval=1.0, n=100):
    new_cmap = colors.LinearSegmentedColormap.from_list(
        'trunc({n},{a:.2f},{b:.2f})'.format(n=cmap.name, a=minval, b=maxval),
        cmap(np.linspace(minval, maxval, n)))
    return new_cmap

def plotField(Xv, Yv, V, title, t=None):
  if t is None:
    plt.quiver(Xv, Yv, V[0](Xv, Yv), V[1](Xv, Yv))
  else:
    plt.quiver(Xv, Yv, V[0](Xv, Yv, t), V[1](Xv, Yv, t))  
  plt.title(title)
  plt.show()
  
def plotScalar(X, Y, U, title, cmap_):
  plt.imshow(U(X,Y), origin="lower", cmap=cmap_, 
             extent=[X[0,0], X[-1, -1], Y[0, 0], Y[-1, -1]])
  plt.title(title)
  plt.colorbar()
  plt.show()
  
def plotIC(X, Y, U, B, W, T, top):
  m = len(X)
  s = m // int(0.1*m)
  plt.figure(figsize=(10, 4))
  X_s, Y_s = X[::s,::s], Y[::s,::s]
  X_t, Y_t = X[::4,::4], Y[::4,::4]
  plt.subplot(1, 2, 1)
  temp = plt.contourf(X, Y, U(X, Y), cmap=plt.cm.jet, alpha=0.8)
  plt.quiver(X_s, Y_s, W[0](X_s, Y_s, 0), W[1](X_s, Y_s, 0))
  plt.colorbar(temp, fraction=0.046, pad=0.04)
  plt.title("Temperature and Wind")
  plt.subplot(1, 2, 2)
  new_cmap = truncate_colormap(plt.cm.gray, 0, 0.5)
  tit = "Fuel"
  #v = np.around(np.linspace(np.min(B(X, Y)), np.max(B(X,Y)), 7), decimals=1)
  fuel = plt.contourf(X, Y, B(X, Y), cmap=plt.cm.Oranges)
  if top is not None:
    topo = plt.contour(X, Y, top(X,Y), vmin=np.min(top(X,Y)), cmap=new_cmap)
    plt.clabel(topo, inline=1, fontsize=10)
  if T is not None:
    plt.quiver(X_t, Y_t, T[0](X_t, Y_t), T[1](X_t, Y_t))
    tit += " and Topography"
  plt.colorbar(fuel, fraction=0.046, pad=0.04)
  plt.title(tit)
  plt.tight_layout()
  plt.show()
  
def plotJCC(t, X, Y, U, B, W, T=None, row=4, col=2, save=False):
  step = 10
  L = len(t)
  W1, W2 = W
  if T is not None:
    T1, T2 = T
  
  up = [None] * row
  bp = [None] * row
  cb = [None] * (row * col)
  
  f, axarr = plt.subplots(4, 2, sharex='col', sharey='row', figsize=(24, 16))
  f.subplots_adjust(left=0.2, right=0.48)
  tim = L // (row - 1)
  X_s = X[::step,::step]
  Y_s = Y[::step,::step]
  r = 0
  vU = np.around(np.linspace(np.min(U), np.max(U), 10), decimals=1)
  vB = np.around(np.linspace(0, np.max(B), 10), decimals=1)
  for i in range(row):  
    tt = i*tim
    if i == (row-1):
      tt = -1
    up[i] = axarr[i, 0].imshow(U[tt], origin='lower', cmap=plt.cm.jet, alpha=0.8, 
      vmin=np.min(U), vmax=np.max(U), extent=[X[-1, 0], X[-1, -1], Y[0, -1], Y[-1, -1]])
    bp[i] = axarr[i, 1].imshow(B[tt], origin='lower', cmap=plt.cm.Oranges, alpha=0.9, 
      vmin=np.min(B), vmax=np.max(B), extent=[X[-1, 0], X[-1, -1], Y[0, -1], Y[-1, -1]])
#    up[i] = axarr[i, 0].contourf(X, Y, U[tt], vU, cmap=plt.cm.jet, alpha=0.9)
#    bp[i] = axarr[i, 1].contourf(X, Y, B[tt], vB,  cmap=plt.cm.Oranges, alpha=0.9)
    axarr[i,0].quiver(X_s, Y_s, W1(X_s, Y_s, t[tt]), W2(X_s, Y_s, t[tt]))
    if T is not None:
      X_t = X[::4,::4]
      Y_t = Y[::4,::4]
      axarr[i,1].quiver(X_t, Y_t, T1(X_t, Y_t), T2(X_t, Y_t))
    cb[r] = plt.colorbar(up[i], ax=axarr[i,0], fraction=0.046, pad=0.04)
    cb[r+1] = plt.colorbar(bp[i], ax=axarr[i,1], fraction=0.046, pad=0.04)
    cb[r].set_label("Temperature", size=14)
    cb[r+1].set_label("Fuel fraction", size=14)
    
    axarr[i,0].tick_params(axis='both', which='major', labelsize=12)
    axarr[i,1].tick_params(axis='both', which='major', labelsize=12)
    axarr[i,0].set_ylabel(r"$y$", fontsize=14)
    if i == (row-1):
      axarr[i,0].set_xlabel(r"$x$", fontsize=14)
      axarr[i,1].set_xlabel(r"$x$", fontsize=14)
      axarr[i,1].tick_params(axis='both', which='major', labelsize=12)
    cb[r].ax.tick_params(labelsize=12)
    cb[r+1].ax.tick_params(labelsize=12)
      
    r += 1
      
  plt.rc('axes', labelsize=12)
  #f.tight_layout()
  
  if save:
    from datetime import datetime
    sec = int(datetime.today().timestamp())
    plt.savefig('experiments/JCC2018/simulations/' + str(sec) + '.pdf', format='pdf', dpi=200, transparent=True, bbox_inches='tight', pad_inches=0)
  else:
    plt.show()

def plotUB(t, X, Y, U, B, V):
  s =  len(X) // int(0.1 * len(X))
  X_s, Y_s = X[::s,::s], Y[::s,::s]
  plt.figure(figsize=(10, 4))
  plt.subplot(1, 2, 1)
  temp = plt.pcolor(X, Y, U[t], cmap=plt.cm.jet, alpha=0.8)
  plt.quiver(X_s, Y_s, V[0](X_s, Y_s, 0), V[1](X_s, Y_s, 0))
  plt.colorbar(temp)
  plt.subplot(1, 2, 2)
  fuel = plt.pcolor(X, Y, B[t], cmap=plt.cm.Oranges)
  plt.colorbar(fuel)
  plt.show()