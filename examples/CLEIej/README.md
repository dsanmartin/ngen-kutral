# Experiments for CLEI Electronic Journal

This folder has 2 subfolders, *CPU* for *NumPy* implementation and *GPU* for *CuPy* implementation. The files in both are:

* ```asensio.py``` performs Asension & Ferragut experiment.
* ```asensio_experiments.py``` performs the computation of 10 experiments per grid size and shows the mean times. 
* ```time_solver.py``` implements IVP solvers.
* ```util.py``` implements utilities for the model.

## Execution

```console
user@host:~ $ python file.py
````