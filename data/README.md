# Data folder

You can put your input/ouput files inside this folder. Framework supports both ```.npy``` and ```.txt``` file formats for fuel and temperature initial conditions, and for wind/terrain vector field. 

## How to use

* Temperature initial condition: Use one file with supported file extensions. File size must match with ```Nx``` and ```Ny``` parameters.
* Fuel initial condition: Use one file with supported file extensions. File size must match with ```Nx``` and ```Ny``` parameters.
* Vector field, terrain + wind: 
	* If you vector field is constant in space, use a supported file with shape ```(Nt + 1) x 2```, one column per component.
	* If your vector field depends of space and time, yo can use a ```.npy``` file with shape ```(Nt + 1) x 2 x Ny x Nx```. Also you can use a folder with ```.txt``` files with the following format: ```V1_{timestep}.txt``` and ```V2_{timestep}.txt``` for each time step, that is for ```0``` to ```Nt```. 

## Example files

[Example](!./input/example/) folder includes some testing files. 
* ```U0.npy``` and ```U0.txt``` for temperature initial condition. Shape of both files is ```128 x 128```.
* ```B0.npy``` and ```B0.txt``` for fuel initial condition. Shape of both files is ```128 x 128```.
* For vector field we include: 
	* ```V.npy``` and ```V.txt``` files with shape ```101 x 2```. 
	* ```VV.npy``` file of shape ```101 x 2 x 128 x 128```.  Also includes ```V``` directory with ```101``` files of shape ```128 x 128``` for both vector components. The data is the same but ```.npy``` can handle differents shapes, while ```.txt``` do not.