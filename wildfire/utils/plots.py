import numpy as np
import matplotlib.pyplot as plt

def UBs(i, t, X, Y, U, B, V):
	s =  len(X) // int(0.1 * len(X))
	X_s, Y_s = X[::s,::s], Y[::s,::s]
	plt.figure(figsize=(10, 4))
	plt.subplot(1, 2, 1)
	#temp = plt.pcolor(X, Y, U[t], cmap=plt.cm.jet, alpha=0.8)
	if i >= 0:
		U = U[i]
		B = B[i]
	temp = plt.imshow(U, origin='lower', cmap=plt.cm.jet, alpha=0.8, 
		vmin=np.min(U), vmax=np.max(U), extent=[X[-1, 0], X[-1, -1], Y[0, -1], Y[-1, -1]])
	if V is not None:
		plt.quiver(X_s, Y_s, V[0](X_s, Y_s, t[i]), V[1](X_s, Y_s, t[i]))
	cb1 = plt.colorbar(temp, fraction=0.046, pad=0.04)
	cb1.set_label("Temperature", size=14)
	#plt.title("Temperature and wind")
	plt.xlabel(r"$x$")
	plt.ylabel(r"$y$")
	plt.subplot(1, 2, 2)
	fuel = plt.pcolor(X, Y, B, cmap=plt.cm.Oranges)
	cb2 = plt.colorbar(fuel, fraction=0.046, pad=0.04)
	cb2.set_label("Fuel Fraction", size=14)
	plt.xlabel(r"$x$")
	plt.ylabel(r"$y$")
	plt.tight_layout()
	plt.show()

def UB(t, X, Y, U, B, V):
	UBs(-1, t, X, Y, U, B, V)
