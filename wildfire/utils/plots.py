import numpy as np
import matplotlib.pyplot as plt

def UBs(i, t, X, Y, U, B, V, type_plot='imshow'):
    rows, cols = X.shape
    # Get sample for quiver
    s_r, s_c =  rows // int(0.1 * rows), cols // int(0.1 * cols)
    X_s, Y_s = X[::s_r,::s_c], Y[::s_r,::s_c]
    # Figure size
    plt.figure(figsize=(10, 4)) 
    # Left plot
    plt.subplot(1, 2, 1)
    # Index -1 is used for last plot option
    if i >= 0:
        U = U[i]
        B = B[i]
    if type_plot == 'contour':
        temp = plt.contourf(X, Y, U, cmap=plt.cm.jet, alpha=0.8, vmin=np.min(U), vmax=np.max(U))
    elif type_plot == 'pcolor':
        temp = plt.pcolor(X, Y, U, cmap=plt.cm.jet, alpha=0.8, vmin=np.min(U), vmax=np.max(U))
    elif type_plot == 'imshow':
        temp = plt.imshow(U, origin='lower', cmap=plt.cm.jet, alpha=0.8,  
            vmin=np.min(U), vmax=np.max(U), extent=[X[-1, 0], X[-1, -1], Y[0, -1], Y[-1, -1]])
    else:
        raise Exception("Type of plot not defined.")
    
    if V is not None:
        if type(V) is np.ndarray:
            if len(V[0].shape) > 2 and len(V[1].shape) > 2: # Check shape of vector field array
                V1, V2 = V[i, 0, ::s_r, ::s_c], V[i, 1, ::s_r, ::s_c]
            else:
                V1, V2 = V[i, 0], V[i, 1]
        else:
            V1, V2 = V(X_s, Y_s, t[i])
        #print(V1, V2)
        plt.quiver(X_s, Y_s, V1, V2)
        #plt.quiver(X_s, Y_s, V[0](X_s, Y_s, t[i]), V[1](X_s, Y_s, t[i]))
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
