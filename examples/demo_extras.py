"""Extra code for demo.

"""
import numpy as np
from wildfire.utils.functions import G
import ipywidgets as widgets

gamma = 1 # Wind effect coefficient...
rm_t = 1 # Set to 0 for no terrain in JCC2019 experiment

## INITIAL CONDITIONS AND VECTOR FIELD ##
# Temperatures
u0_af2002  = lambda x, y: 1 * G(x-50, y-50, 100)
u0_jcc2018 = lambda x, y: 6 * G(x-20, y-20, 20)
u0_jcc2019 = lambda x, y: 7 * G(x-20, y-20, 20) + 4 * G(x-80, y-70, 20) + 4 * G(x-20, y-35, 50)

# Winds
w1_jcc2019 = lambda x, y, t: gamma * np.cos(np.pi/4 + x*0 + t * 0.025) 
w2_jcc2019 = lambda x, y, t: gamma * np.sin(np.pi/4 + x*0 + t * 0.025) 

# Terrain effect 
T = lambda x, y: 1.5 * (3 * G(x-45, y-45, 40) + 2 * G(x-30, y-30, 60) + 3 * G(x-70, y-70, 60) + 2 * G(x-20, y-70, 70))
Tx = lambda x, y: -2 * 1.5 * ( 3 * (x-45) * G(x-45, y-45, 40) / 40 + 2 * (x-30) * G(x-30, y-30, 60) / 60 + 3 * (x-70) * G(x-70, y-70, 60) / 60 + 2 * (x-20) * G(x-20, y-70, 70) / 70)
Ty = lambda x, y: -2 * 1.5 * ( 3 * (y-45) * G(x-45, y-45, 40) / 40 + 2 * (y-30) * G(x-30, y-30, 60) / 60 + 3 * (y-70) * G(x-70, y-70, 60) / 60 + 2 * (y-70) * G(x-20, y-70, 70) / 70) 
TT = lambda x, y: (Tx(x, y), Ty(x, y))

# Vector fields
v1_af2002  = lambda x, y, t: 300
v2_af2002  = lambda x, y, t: 300
v1_jcc2018 = lambda x, y, t: gamma * np.cos(np.pi/4)
v2_jcc2018 = lambda x, y, t: gamma * np.sin(np.pi/4)
v1_jcc2019 = lambda x, y, t: w1_jcc2019(x, y, t) + rm_t * Tx(x, y)
v2_jcc2019 = lambda x, y, t: w2_jcc2019(x, y, t) + rm_t * Ty(x, y)

V_af2002  = lambda x, y, t: (v1_af2002(x, y, t), v2_af2002(x, y, t))
V_jcc2018 = lambda x, y, t: (v1_jcc2018(x, y, t), v2_jcc2018(x, y, t))
V_jcc2019 = lambda x, y, t: (v1_jcc2019(x, y, t), v2_jcc2019(x, y, t))

# Fuels

def b0_bc(B):
    rows, cols = B.shape
    B[ 0,:] = np.zeros(cols)
    B[-1,:] = np.zeros(cols)
    B[:, 0] = np.zeros(rows)
    B[:,-1] = np.zeros(rows)
    return B

def b0_jcc2019(x, y):
    br = lambda x, y: 1 + x * 0 
    return b0_bc(br(x, y))

def b0_af2002(x, y):
    np.random.seed(777)
    br = lambda x, y: np.round(np.random.uniform(size=(x.shape)), decimals=2)
    return b0_bc(br(x, y))
    

## WIDGETS EXTRAS ##
# Experiments
experiment_dp =  widgets.Dropdown(
    options=[('Asensio & Ferragut 2002', 'af2002'), ('JCC 2018', 'jcc2018'), ('JCC 2019', 'jcc2019')], 
    value='jcc2018', 
    description="Experiment"
)

# Widget to select space method
space_method_dp = widgets.Dropdown(
    options=[('Finite Difference', 'fd'), ('Fast Fourier Transform', 'fft')], 
    value='fd', 
    description="Space Method"
)

# Finite difference accuracy widget
space_acc_dp = widgets.Dropdown(
    options=[2, 4], 
    value=2, 
    description="Finite Difference Accuracy"
)

# Widget to define number of x nodes
space_nx_text = widgets.BoundedIntText(
    value=128,
    min=64,
    max=512,
    step=32,
    description=r"$N_x$:",
    disabled=False
)

# Widget to define number of y nodes
space_ny_text = widgets.BoundedIntText(
    value=128,
    min=64,
    max=512,
    step=32,
    description=r"$N_y$:",
    disabled=False
)

# Widget to select time method
time_method_dp_1 = widgets.Dropdown(
    options=[('Euler Method', 'Euler'), ('Runge-Kutta of 4th order', 'RK4')], 
    value='RK4', 
    description="Time Method"
)

time_method_dp_2 = widgets.Dropdown(
    options=['Euler', 'RK4', 'RK23', 'RK45', 'DOP853', 'Radau', 'BDF', 'LSODA'], 
    value='RK4', 
    description="Time Method"
)

# Widget to define number of t nodes
time_nt_text = widgets.BoundedIntText(
    value=100,
    min=100,
    max=3000,
    step=50,
    description=r"$N_t$:",
    disabled=False
)

# Checkbox to select last approximation
time_last_cb = widgets.Checkbox(
        value=False,
        description='Last approximation:',
        disabled=False,
        indent=False
)
out = widgets.Output()