import numpy as np
#import matplotlib
#matplotlib.use('agg')
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import datetime, pathlib

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
  
def plotIC(X, Y, U, B, W, T, top, save=False):
  # Domain
  m = len(X)
  s = m // int(0.1*m)
  X_s, Y_s = X[::s,::s], Y[::s,::s]
  X_t, Y_t = X[::4,::4], Y[::4,::4]
  f, axarr = plt.subplots(1, 2, sharex='col', sharey='row', figsize=(10, 4))
  plot_name = "Temperature"
  temp = axarr[0].contourf(X, Y, U(X, Y), cmap=plt.cm.jet, alpha=0.8)
  axarr[0].set_xlabel(r"$x$")
  axarr[0].set_ylabel(r"$y$")
  plt.colorbar(temp, fraction=0.046, pad=0.04, ax=axarr[0])
  if W is not None:
    axarr[0].quiver(X_s, Y_s, W[0](X_s, Y_s, 0), W[1](X_s, Y_s, 0))
    plot_name += " and Wind"
  axarr[0].set_title(plot_name)
  new_cmap = truncate_colormap(plt.cm.gray, 0, 0.5)
  tit = "Fuel"
  #v = np.around(np.linspace(np.min(B(X, Y)), np.max(B(X,Y)), 7), decimals=1)
  fuel = axarr[1].contourf(X, Y, B(X, Y), cmap=plt.cm.Oranges, alpha=1 if T is None else 0.5)
  if top is not None:
    topo = axarr[1].contour(X, Y, top(X,Y), vmin=np.min(top(X,Y)), cmap=new_cmap)
    plt.clabel(topo, inline=1, fontsize=10)
  if T is not None:
    axarr[1].quiver(X_t, Y_t, T[0](X_t, Y_t), T[1](X_t, Y_t))
    tit += " and Topography"
  plt.colorbar(fuel, fraction=0.046, pad=0.04, ax=axarr[1])
  axarr[1].set_title(tit)
  axarr[1].set_xlabel(r"$x$")
  
  if save:
    pass
  else:
    plt.show()
  
def plotJCC(t, X, Y, U, B, W, T=None, row=4, col=2, save=False):
  step = 10
  L = len(t)
  if type(W) is tuple:
    W1, W2 = W
  if T is not None:
    T1, T2 = T
  
  up = [None] * row
  bp = [None] * row
  cb = [None] * (row * col)
  
  f, axarr = plt.subplots(row, col, sharex='col', sharey='row', figsize=(24, 16))
  #f.subplots_adjust(left=0.2, right=0.48)
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
    if type(W) is tuple:
      axarr[i,0].quiver(X_s, Y_s, W1(X_s, Y_s, t[tt]), W2(X_s, Y_s, t[tt]))
    else:
      axarr[i,0].quiver(X_s, Y_s, W[tt, 0]*np.ones_like(X_s), W[tt, 1]*np.ones_like(Y_s))
      
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
  f.tight_layout()
  
  if save:
    from datetime import datetime
    sec = int(datetime.today().timestamp())
    plt.savefig('experiments/JCC2018/simulations/' + str(sec) + '.pdf', format='pdf', dpi=200, transparent=True, bbox_inches='tight', pad_inches=0)
  else:
    plt.show()
    
def plotJCCCols(t, X, Y, U, B, W, T=None, row=2, col=4, save=False):
  step = 10
  L = len(t)
  if type(W) is tuple:
    W1, W2 = W
  if T is not None:
    T1, T2 = T
  
  up = [None] * col
  bp = [None] * col
  cb = [None] * (row * col)
  
  f, axarr = plt.subplots(row, col, sharex='col', sharey='row', figsize=(23, 10))
  #f.subplots_adjust(left=0.2, right=0.48)
  tim = L // (col - 1)
  X_s = X[::step,::step]
  Y_s = Y[::step,::step]
  r = 0
  vU = np.around(np.linspace(np.min(U), np.max(U), 10), decimals=1)
  vB = np.around(np.linspace(0, np.max(B), 10), decimals=1)
  for i in range(col):  
    tt = i*tim
    if i == (col-1):
      tt = -1
    up[i] = axarr[0, i].imshow(U[tt], origin='lower', cmap=plt.cm.jet, alpha=0.8, 
      vmin=np.min(U), vmax=np.max(U), extent=[X[-1, 0], X[-1, -1], Y[0, -1], Y[-1, -1]])
    bp[i] = axarr[1, i].imshow(B[tt], origin='lower', cmap=plt.cm.Oranges, alpha=0.9, 
      vmin=np.min(B), vmax=np.max(B), extent=[X[-1, 0], X[-1, -1], Y[0, -1], Y[-1, -1]])
#    up[i] = axarr[i, 0].contourf(X, Y, U[tt], vU, cmap=plt.cm.jet, alpha=0.9)
#    bp[i] = axarr[i, 1].contourf(X, Y, B[tt], vB,  cmap=plt.cm.Oranges, alpha=0.9)
    if type(W) is tuple:
      axarr[0,i].quiver(X_s, Y_s, W1(X_s, Y_s, t[tt]), W2(X_s, Y_s, t[tt]))
    else:
      axarr[0,i].quiver(X_s, Y_s, W[tt, 0]*np.ones_like(X_s), W[tt, 1]*np.ones_like(Y_s))
      
    if T is not None:
      X_t = X[::4,::4]
      Y_t = Y[::4,::4]
      axarr[1,i].quiver(X_t, Y_t, T1(X_t, Y_t), T2(X_t, Y_t))
#    cb[r] = plt.colorbar(up[i], ax=axarr[0,i], fraction=0.046, pad=0.04)
#    cb[r+1] = plt.colorbar(bp[i], ax=axarr[1,i], fraction=0.046, pad=0.04)
#    cb[r].set_label("Temperature", size=14)
#    cb[r+1].set_label("Fuel fraction", size=14)
    
    axarr[0,i].tick_params(axis='both', which='major', labelsize=16)
    axarr[1,i].tick_params(axis='both', which='major', labelsize=16)
    axarr[1,i].set_xlabel(r"$x$", fontsize=18)
    if i == 0:
      axarr[0,i].set_ylabel(r"$y$", fontsize=18)
      axarr[1,i].set_ylabel(r"$y$", fontsize=18)
      axarr[1,i].tick_params(axis='both', which='major', labelsize=16)
    #cb[r].ax.tick_params(labelsize=12)
    #cb[r+1].ax.tick_params(labelsize=12)
      
    r += 1
  cb1 = plt.colorbar(up[-1], ax=axarr[0,-1], fraction=0.046, pad=0.04)
  cb2 = plt.colorbar(bp[-1], ax=axarr[1,-1], fraction=0.046, pad=0.04)
  cb1.set_label("Temperatura", size=18)
  cb2.set_label("Fracción de Combustible", size=18)
  cb1.ax.tick_params(labelsize=16)
  cb2.ax.tick_params(labelsize=16)
      
  plt.rc('axes', labelsize=12)
  f.tight_layout()
  
  if save:
    from datetime import datetime
    sec = int(datetime.today().timestamp())
    plt.savefig('experiments/JCC2018/simulations/' + str(sec) + '.pdf', format='pdf', dpi=200, transparent=True, bbox_inches='tight', pad_inches=0)
  else:
    plt.show()

def plotUB(i, t, X, Y, U, B, V, save=False, DIR_BASE=""):
  s =  len(X) // int(0.1 * len(X))
  X_s, Y_s = X[::s,::s], Y[::s,::s]
  plt.figure(figsize=(10, 4))
  plt.subplot(1, 2, 1)
  #temp = plt.pcolor(X, Y, U[t], cmap=plt.cm.jet, alpha=0.8)
  temp = plt.imshow(U[i], origin='lower', cmap=plt.cm.jet, alpha=0.8, 
    vmin=np.min(U), vmax=np.max(U), extent=[X[-1, 0], X[-1, -1], Y[0, -1], Y[-1, -1]])
  plt.quiver(X_s, Y_s, V[0](X_s, Y_s, t[i]), V[1](X_s, Y_s, t[i]))
  cb1 = plt.colorbar(temp, fraction=0.046, pad=0.04)
  cb1.set_label("Temperature", size=14)
  #plt.title("Temperature and wind")
  plt.xlabel(r"$x$")
  plt.ylabel(r"$y$")
  plt.subplot(1, 2, 2)
  fuel = plt.pcolor(X, Y, B[i], cmap=plt.cm.Oranges)
  cb2 = plt.colorbar(fuel, fraction=0.046, pad=0.04)
  cb2.set_label("Fuel Fraction", size=14)
  plt.xlabel(r"$x$")
  plt.ylabel(r"$y$")
  plt.tight_layout()
  
  if save:
    img_name = int(i/100)
    if img_name < 10:
      img_name = '0' + str(img_name)
    else:
      img_name = str(img_name)
      
    plt.savefig(DIR_BASE + img_name + '.png', format='png', dpi=200, transparent=True, bbox_inches='tight', pad_inches=0)
  else:
    plt.show()

def plotUBs(i, t, X, Y, U, B, V):
  s =  len(X) // int(0.1 * len(X))
  X_s, Y_s = X[::s,::s], Y[::s,::s]
  plt.figure(figsize=(10, 4))
  plt.subplot(1, 2, 1)
  #temp = plt.pcolor(X, Y, U[t], cmap=plt.cm.jet, alpha=0.8)
  temp = plt.imshow(U[i], origin='lower', cmap=plt.cm.jet, alpha=0.8, 
    vmin=np.min(U), vmax=np.max(U), extent=[X[-1, 0], X[-1, -1], Y[0, -1], Y[-1, -1]])
  if V is not None:
    plt.quiver(X_s, Y_s, V[0](X_s, Y_s, t[i]), V[1](X_s, Y_s, t[i]))
  cb1 = plt.colorbar(temp, fraction=0.046, pad=0.04)
  cb1.set_label("Temperature", size=14)
  #plt.title("Temperature and wind")
  plt.xlabel(r"$x$")
  plt.ylabel(r"$y$")
  plt.subplot(1, 2, 2)
  fuel = plt.pcolor(X, Y, B[i], cmap=plt.cm.Oranges)
  cb2 = plt.colorbar(fuel, fraction=0.046, pad=0.04)
  cb2.set_label("Fuel Fraction", size=14)
  plt.xlabel(r"$x$")
  plt.ylabel(r"$y$")
  plt.tight_layout()
  plt.show()
    
def plotComplete(t, X, Y, U, B, V, save=False):
  now = datetime.datetime.now() 
  SIM_NAME = now.strftime("%Y%m%d%H%M%S")
  DIR_BASE = "experiments/simulations/" + SIM_NAME + "/"
  pathlib.Path(DIR_BASE).mkdir(parents=True, exist_ok=True)
  
  dir_base = ""
  if save: dir_base = DIR_BASE
  
  for tt in range(len(U)):
    if tt % 100 == 0:
      plotUB(tt, t, X, Y, U, B, V, save, dir_base)