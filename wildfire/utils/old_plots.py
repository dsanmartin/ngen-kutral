import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as colors

## OLD PLOTS... ##
# JCC 2018
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
		axarr[0].quiver(X_s, Y_s, W(X_s, Y_s, 0)[0], W(X_s, Y_s, 0)[1])
		#axarr[0].quiver(X_s, Y_s, W[0](X_s, Y_s, 0), W[1](X_s, Y_s, 0))
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
		axarr[1].quiver(X_t, Y_t, T(X_t, Y_t)[0], T(X_t, Y_t)[1])
		#axarr[1].quiver(X_t, Y_t, T[0](X_t, Y_t), T[1](X_t, Y_t))
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
		if type(W) is np.ndarray:
			axarr[i,0].quiver(X_s, Y_s, W[tt, 0]*np.ones_like(X_s), W[tt, 1]*np.ones_like(Y_s))
		else:
			axarr[i,0].quiver(X_s, Y_s, W(X_s, Y_s, t[tt])[0], W(X_s, Y_s, t[tt])[1])
		# if type(W) is tuple:
		# 	axarr[i,0].quiver(X_s, Y_s, W1(X_s, Y_s, t[tt]), W2(X_s, Y_s, t[tt]))
		# else:
		# 	axarr[i,0].quiver(X_s, Y_s, W[tt, 0]*np.ones_like(X_s), W[tt, 1]*np.ones_like(Y_s))
			
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

def truncate_colormap(cmap, minval=0.0, maxval=1.0, n=100):
	new_cmap = colors.LinearSegmentedColormap.from_list(
			'trunc({n},{a:.2f},{b:.2f})'.format(n=cmap.name, a=minval, b=maxval),
			cmap(np.linspace(minval, maxval, n)))
	return new_cmap

# JCC 2019
def spatialComplexity(times, c=(1, 1)):
  rows, cols = times.shape
  N = times[:,0]
  fd_times = times[:,1]
  fft_times = times[:,2]
  #for i in range(1, cols)
 # plt.plot(N, fd_times, 'r-x', label="FD")
  plt.plot(N, fft_times, 'b-o', label="FFT")
  #plt.plot(NN, 1e-6*NN**3, label=r"$O(NN^3)$")
  plt.plot(N, c[0]*N**3, 'r--', label=r"$O(N^3)$", linewidth=2)
  
  plt.plot(N, c[1]*N**2*np.log(N), 'k--', label=r"$O(N^2\log(N))$", linewidth=2)
  #plt.plot(N, c[0]*N**2, 'g--', label=r"$O(N^2)$", linewidth=2)
  plt.xscale('log', basex=2)
  plt.yscale('log')
  plt.xlabel("N")
  plt.ylabel("Time [s]")
  plt.grid(True)
  plt.legend()
  plt.show()

def spatialConvergence(errors, c=1):
  rows, cols = errors.shape
  dxs = errors[:,0]
  fd_errs = errors[:,1]
  fft_errs = errors[:,2]
  plt.plot(dxs, fd_errs, 'r-x', label="FD")
  plt.plot(dxs, fft_errs, 'b-o', label="FFT")
  plt.plot(dxs, c*dxs**2, 'r--', label=r"$O(h^2)$")
  #plt.plot(dxs, 1e2*dxs**4, 'g:', label=r"$O(h^4)$")
  plt.grid(True)
  plt.xscale('log', basex=2)
  plt.yscale('log')
  plt.xlabel(r"$h=\Delta x = \Delta y$")
  plt.ylabel("Error")
  plt.legend()
  plt.show()
	
def timeConvergence(errors):
  rows, cols = errors.shape
  dts = errors[:,0]
  eul_errs = errors[:,1]
  rk4_errs = errors[:,2]
  plt.plot(dts, eul_errs, 'r-x', label="Euler")
  plt.plot(dts, rk4_errs, 'b-o', label="RK4")
  plt.plot(dts, 2e-2*(dts**4), 'g:', label=r"$O(\Delta t^4)$", linewidth=2)
  plt.plot(dts, 1e2*(dts**2), 'k--', label=r"$O(\Delta t^2)$", linewidth=2)
  plt.grid(True)
  plt.xscale('log', basex=2)
  plt.yscale('log')
  plt.xlabel(r"$\Delta t$")
  plt.ylabel("Error")
  plt.legend()
  plt.show()

def timeComplexity(times):
  rows, cols = times.shape
  L = times[:,0]
  eul_times = times[:,1]
  rk4_times = times[:,2]
  plt.plot(L, eul_times, 'r-x', label="Euler")
  plt.plot(L, rk4_times, 'b-o', label="RK4")
  plt.plot(L, 3e-2*L, 'm--', label=r"$O(L)$")
  plt.xscale('log', basex=2)
  plt.yscale('log')
  plt.xlabel('L')
  plt.ylabel("Time [s]")
  plt.legend()
  plt.grid(True)
  plt.show()

def plotCompare(x_lim, y_lim, U1, B1, U2, B2, m1, m2):
  plt.figure(figsize=(10,6))
  extent_ = [x_lim[0], x_lim[1], y_lim[0], y_lim[1]]
  plt.subplot(2, 3, 1)
  plt.imshow(U1, origin="lower", cmap=plt.cm.jet, extent=extent_)
  plt.title(m1)
  plt.colorbar()
  plt.subplot(2, 3, 2)
  plt.imshow(U2, origin="lower", cmap=plt.cm.jet, extent=extent_)
  plt.title(m2)
  plt.colorbar()
  plt.subplot(2, 3, 3)
  plt.imshow(np.abs(U1 - U2), origin="lower", extent=extent_)
  plt.colorbar()
  plt.title("Absolute Error")
  plt.subplot(2, 3, 4)
  plt.imshow(B1, origin="lower", cmap=plt.cm.Oranges, extent=extent_)
  plt.colorbar()
  plt.subplot(2, 3, 5)
  plt.imshow(B2, origin="lower", cmap=plt.cm.Oranges, extent=extent_)
  plt.colorbar()
  plt.subplot(2, 3, 6)
  plt.imshow(np.abs(B1 - B2), origin="lower", extent=extent_)
  plt.colorbar()
  plt.tight_layout()
  plt.show()