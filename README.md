# Ngen Kütral
## Open Source Framework for Chilean Wildfire Spreading

This repository includes:
* Source code of Master's degree thesis: *Open Source Framework for Chilean Wildfire Spreading and Effects Analysis*. 
* Experiments of:
	- **2018 37th International Conference of the Chilean Computer Science Society** paper *Ngen Kütral: Toward an Open Source Framework for Chilean Wildfire Spreading* by Daniel San Martin & Claudio E. Torres. https://doi.org/10.1109/SCCC.2018.8705159.
	- **2019 38th International Conference of the Chilean Computer Science Society** paper *Exploring a Spectral Numerical Algorithm for Solving a Wildfire Mathematical Model* by Daniel San Martin & Claudio E. Torres. https://doi.org/10.1109/SCCC49216.2019.8966412.

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
usage: main.py [-h] [-kap K] [-eps E] [-upc P] [-alp A] [-qrh Q]
               [-xlim x_min x_max] [-ylim y_min y_max] [-tlim t_min t_max] -sm
               SM -Nx Nx -Ny Ny -tm TM -Nt Nt -u0 U0 -b0 B0 [-vf V] [-acc ACC]
               [-sps S] [-lst LST] [-plt PLT]

Create and execute a wildfire simulation.

optional arguments:
  -h, --help            show this help message and exit
  -kap K, --kappa K     kappa parameter (default: 1e-1)
  -eps E, --epsilon E   epsilon parameter (default: 3e-1)
  -upc P, --phase P     phase change threshold parameter (default: 3)
  -alp A, --alpha A     alpha parameter (default: 1e-3)
  -qrh Q, --reaction Q  alpha parameter (default: 1.0)
  -xlim x_min x_max, --xlimits x_min x_max
                        x domain limits (default: [0, 90])
  -ylim y_min y_max, --ylimits y_min y_max
                        y domain limits (default: [0, 90])
  -tlim t_min t_max, --tlimits t_min t_max
                        t domain limits (default: [0, 30])
  -sm SM, --space SM    Space method approximation
  -Nx Nx, --xnodes Nx   Number of nodes in x
  -Ny Ny, --ynodes Ny   Number of nodes in y
  -tm TM, --time TM     Time method approximation
  -Nt Nt, --tnodes Nt   Number of nodes in t
  -u0 U0, --initial-temperature U0
                        Initial temperature file. Only .txt and .npy
                        supported.
  -b0 B0, --initial-fuel B0
                        Initial fuel file. Only .txt and .npy supported.
  -vf V, --vector-field V
                        Vector Field. Only .txt and .npy supported.
  -acc ACC, --accuracy ACC
                        Finite difference accuracy (default: 2)
  -sps S, --sparse S    Finite difference sparse matrices (default: False)
  -lst LST, --last LST  Only last approximation (default: True)
  -plt PLT, --plot PLT  Plot result (default: False)
```


### Examples and experiments

You can check the following examples of usage and paper experiments:
* [Examples](./examples/)
* [Experiments of JCC 2018](./examples/JCC2018/)
* [Experiments of JCC 2019](./examples/JCC2019/)