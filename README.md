# Ngen Kütral
## Open Source Framework for Chilean Wildfire Spreading

Source code of Master's degree thesis 

Also includes JCC2018 experiments for submitted paper *Ngen Kütral: Toward an Open Source Framework for Chilean Wildfire Spreading* by Daniel San Martin & Claudio E. Torres

### Prerequisites

To use our framework you must install

```
Python >= 3.6.5
Numpy >= 1.13.3
Scipy >= 1.1.0
Matplotlib >= 2.2.2
```

### Installing

Clone the repository at the directory you want
```console
user@host:~$ git clone https://github.com/dsanmartin/ngen-kutral.git

```

Add the following line to your ~/.profile, ~/.bash_profile or ~/.bashrc file according to your SO configuration.
```
export PYTHONPATH=$PYTHONPATH:/path/you/put/the/repository/ngen-kutral/widlfire
```
### Run experiments
Execute the following command inside repository directory
```console
user@host:~$ python experiments/JCC2018/experiment_you_want.py
```