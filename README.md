# Ngen Kütral
## Open Source Framework for Chilean Wildfire Spreading

This repository includes:
* Source code of Master's degree thesis: *Open Source Framework for Chilean Wildfire Spreading and Effects Analysis*. 
* Experiments of:
	- **2018 37th International Conference of the Chilean Computer Science Society** paper *Ngen Kütral: Toward an Open Source Framework for Chilean Wildfire Spreading* by Daniel San Martin & Claudio E. Torres. https://doi.org/10.1109/SCCC.2018.8705159.
	- **2019 38th International Conference of the Chilean Computer Science Society** paper *Exploring a Spectral Numerical Algorithm for Solving a Wildfire Mathematical Model* by Daniel San Martin & Claudio E. Torres. https://doi.org/10.1109/SCCC49216.2019.8966412.
    - **CLEI Electronic Journal** paper *2D Simplified Wildfire Spreading Model in Python: From NumPy to CuPy* by Daniel San Martin & Claudio E. Torres. (Work in progress)

## Prerequisites

To use our framework you must install

```
Python >= 3.6.5
Numpy >= 1.13.3
Scipy >= 1.1.0
Matplotlib >= 2.2.2
```

## Installing

Clone the repository at the directory you want
```console
user@host:~ $ git clone https://github.com/dsanmartin/ngen-kutral.git

```

Add the following line to your ~/.profile, ~/.bash_profile or ~/.bashrc file according to your SO configuration.
```
export PYTHONPATH=$PYTHONPATH:/path/you/put/the/repository/ngen-kutral/
```

## How to use

1. As module:
```python
import wildfire
```

2. Using a wrapper script inside ```bin``` [folder](./bin):
```console
usage: main.py -sm SM -Nx NX -Ny NY -tm TM -Nt NT [-h] [-k K] [-e E] [-p P]
               [-a A] [-q Q] [-x XMIN XMAX] [-y YMIN YMAX] [-t TMIN TMAX]
               [-u0 U0] [-b0 B0] [-vf VF] [-w W] [-T Tx Ty] [-acc ACC]
               [-sps S] [-lst LST] [-plt PLT]

Create and execute a wildfire simulation.

required arguments:
  -sm SM, --space SM    Space method approximation, FD (Finite Difference) or
                        FFT (Fast Fourier Transform).
  -Nx NX, --xnodes NX   Number of nodes in x.
  -Ny NY, --ynodes NY   Number of nodes in y.
  -tm TM, --time TM     Time method approximation, Euler or RK4.
  -Nt NT, --tnodes NT   Number of nodes in t.

optional arguments:
  -h, --help            Show this help message and exit.
  -k K, --kappa K       Kappa parameter (default: 0.1).
  -e E, --epsilon E     Epsilon parameter (default: 0.3).
  -p P, --phase P       Phase change threshold parameter (default: 3.0).
  -a A, --alpha A       Alpha parameter (default: 1e-3).
  -q Q, --reaction Q    Reaction heat coefficient (default: 1.0).
  -x XMIN XMAX, --xlimits XMIN XMAX
                        x domain limits (default: [0, 90]).
  -y YMIN YMAX, --ylimits YMIN YMAX
                        y domain limits (default: [0, 90]).
  -t TMIN TMAX, --tlimits TMIN TMAX
                        t domain limits (default: [0, 30]).
  -u0 U0, --initial-temperature U0
                        Initial temperature file. Only .txt and .npy supported
                        (default lambda testing function).
  -b0 B0, --initial-fuel B0
                        Initial fuel file. Only .txt and .npy supported
                        (default lambda testing function).
  -vf VF, --vector-field VF
                        Vector Field. Only .txt and .npy supported (default
                        lambda testing function).
  -w W, --wind W        Wind component. Only .txt and .npy supported (default
                        lambda testing function).
  -T Tx Ty, --terrain Tx Ty
                        Topography gradient effect. Only .txt and .npy
                        supported (default no topography effect).
  -acc ACC, --accuracy ACC
                        Finite difference accuracy (default: 2).
  -sps S, --sparse S    Finite difference sparse matrices (default: 0).
  -lst LST, --last LST  Only last approximation (default: 1).
  -plt PLT, --plot PLT  Plot result (default: 0).
```


### Examples and experiments

You can check the following examples of usage and paper experiments:
* [Examples](./examples/)
* [Experiments of JCC 2018](./examples/JCC2018/)
* [Experiments of JCC 2019](./examples/JCC2019/)
* [Experiments of CLEIej](./examples/CLEIej/)