import os
import glob
import pathlib
import json
import inspect
import numpy as np
from datetime import datetime

def simulation():
    # Simulation id
    SIM_ID = datetime.now().strftime("%Y%m%d%H%M%S")
    DIR_BASE = "./experiments/JCC2019/data/" + SIM_ID + "/"
    pathlib.Path(DIR_BASE).mkdir(parents=True, exist_ok=True)
    return SIM_ID, DIR_BASE

def saveParameters(DIR_BASE, parameters):
    param = dict(parameters)
    param['M'] = int(parameters['M'])
    param['N'] = int(parameters['N'])
    param['L'] = int(parameters['L'])
    param['b0'] = inspect.getsourcelines(parameters['b0'])[0][0].strip("['\n']")
    param['u0'] = inspect.getsourcelines(parameters['u0'])[0][0].strip("['\n']")
    
    # Vector field
    param['v'] = (
        inspect.getsourcelines(param['v'][0])[0][0].strip("['\n']"),
        inspect.getsourcelines(param['v'][1])[0][0].strip("['\n']")
    )
  
    with open(DIR_BASE + 'parameters.json', 'w') as fp:
        json.dump(param, fp)

def saveSimulation(DIR_BASE, U, B):
    np.save(DIR_BASE + 'U', U)
    np.save(DIR_BASE + 'B', B)
  
def saveSimulationCSV(DIR_BASE, parameters, U, B, k):
    M, N = parameters['N'], parameters['M']
    L = parameters['L']
    xa, xb = parameters['x_lim']
    ya, yb = parameters['y_lim']
    ta, tb = parameters['t_lim']
    x = np.linspace(xa, xb, N)
    y = np.linspace(ya, yb, M)
    t = np.linspace(ta, tb, L + 1)
    X, Y = np.meshgrid(x, y)
    UU = np.zeros((2 ** (M * N), 3))
    BB = np.zeros_like(UU)
    UU[:,0] = X.flatten()
    UU[:,1] = Y.flatten()
    UU[:,2] = U[k].flatten() 
    BB[:,0] = X.flatten()
    BB[:,1] = Y.flatten()
    BB[:,2] = B[k].flatten() 
    np.savetxt('U' + str(int(t[k]))+ '.csv', UU, fmt="%.8f")
    np.savetxt('B' + str(int(t[k]))+ '.csv', BB, fmt="%.8f")
  
def saveTimeError(DIR_BASE, times, errors):
    np.save(DIR_BASE + 'times', times)
    np.save(DIR_BASE + 'errors', errors)
  
def openFile(dir):
    """Function to handle data files.

    Parameters
    ----------
    dir : str
        Data file path.

    Returns
    -------
    array_like
        Data.

    Raises
    ------
    Exception
        Error if file format or path is not supported.
    """
    # If path is a directory
    if os.path.isdir(dir):
        # Check file sizes from first file
        Nt = len(glob.glob(dir + "/1_*.txt")) # Nt
        tmp = np.loadtxt(dir + "/1_0.txt") 
        Ny, Nx = tmp.shape
        # Output array
        V = np.zeros((Nt, 2, Ny, Nx))
        for n in range(Nt):
            V1 = np.loadtxt("{0}/1_{1}.txt".format(dir, n))
            V2 = np.loadtxt("{0}/2_{1}.txt".format(dir, n))
            V[n] = V1, V2
        return V
    # If path is a file
    elif os.path.isfile(dir):
        if '.npy' in dir: # Numpy data file
            return np.load(dir)
        elif '.txt' in dir: # Plain text data file
            return np.loadtxt(dir)
        else:
            raise Exception("File extension not supported.")
    else:
        raise Exception("Path is not supported.")


def openVectorWT(w_dir, t_dir):
    """Open vector field with independent directories.

    Parameters
    ----------
    w_dir : str
        Wind effect filepath.
    t_dir : tuple (str, str)
        Topography effect filepaths.
    """
    tx_dir, ty_dir = t_dir
    W = openFile(w_dir)
    Tx = openFile(tx_dir)
    Ty = openFile(ty_dir)
    assert Tx.shape == Ty.shape, "Tx and Ty must have same dimensions."
    Nt, Nc = W.shape
    Ny, Nx = Tx.shape 
    V = np.zeros((Nt, 2, Ny, Nx))
    for n in range(Nt):
        V[n, 0] = Tx + W[n, 0]
        V[n, 1] = Ty + W[n, 1]
    return V