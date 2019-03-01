timport os 
import matplotlib.pyplot as plt
import numpy as np
#%%

DIR_BLOCK = "/Volumes/My Passport/Data/Thesis/cluster_experiments/output_gpu_20190224/"
DIR_ALL = "/Volumes/My Passport/Data/Thesis/cluster_experiments/output_all_512DB/"
DIR_PY = "/Volumes/My Passport/Data/Thesis/cluster_experiments/output_20190224_2_py/"

b_folders = [dI for dI in os.listdir(DIR_BLOCK) if os.path.isdir(os.path.join(DIR_BLOCK, dI))]
a_folders = [dI for dI in os.listdir(DIR_ALL) if os.path.isdir(os.path.join(DIR_ALL, dI))]
p_folders = [dI for dI in os.listdir(DIR_PY) if os.path.isdir(os.path.join(DIR_PY, dI))]


B_L_500 = list()
B_L_1000 = list()
B_L_1500 = list()
B_L_2000 = list()
A_L_500 = list()
A_L_1000 = list()
A_L_1500 = list()
A_L_2000 = list()
P_L_500 = list()
P_L_1000 = list()
P_L_1500 = list()
P_L_2000 = list()


for sim_dir in b_folders:
  with open(DIR_BLOCK + sim_dir + "/log.txt") as f:
    lines = f.readlines()
    n_sims = int(lines[1].split()[-1])
    L = int(lines[22].split()[-1])
    time = float(lines[25].split()[-2])
    if L == 500:
      B_L_500.append((n_sims, time))
    elif L == 1000:
      B_L_1000.append((n_sims, time))
    elif L == 1500:
      B_L_1500.append((n_sims, time))
    elif L == 2000:
      B_L_2000.append((n_sims, time))
    
      
for sim_dir in a_folders:
  with open(DIR_ALL + sim_dir + "/log.txt") as f:
    lines = f.readlines()
    n_sims = int(lines[1].split()[-1])
    L = int(lines[22].split()[-1])
    time = float(lines[25].split()[-2])
    if L == 500:
      A_L_500.append((n_sims, time))
    elif L == 1000:
      A_L_1000.append((n_sims, time))
    elif L == 1500:
      A_L_1500.append((n_sims, time))
    elif L == 2000:
      A_L_2000.append((n_sims, time))
      
for sim_dir in p_folders:
  with open(DIR_PY + sim_dir + "/log.txt") as f:
    lines = f.readlines()
    n_sims = int(lines[1].split()[-1])
    L = int(lines[22].split()[-1])
    time = float(lines[25].split()[-2])
    if L == 500:
      P_L_500.append((n_sims, time))
    elif L == 1000:
      P_L_1000.append((n_sims, time))
    elif L == 1500:
      P_L_1500.append((n_sims, time))
    elif L == 2000:
      P_L_2000.append((n_sims, time))
      
B_L_500 = np.array(sorted(B_L_500))[:,1]
B_L_1000 = np.array(sorted(B_L_1000))[:,1]
B_L_1500 = np.array(sorted(B_L_1500))[:,1]
B_L_2000 = np.array(sorted(B_L_2000))[:,1]

A_L_500 = np.array(sorted(A_L_500))[:,1]
A_L_1000 = np.array(sorted(A_L_1000))[:,1]
A_L_1500 = np.array(sorted(A_L_1500))[:,1]
A_L_2000 = np.array(sorted(A_L_2000))[:,1]

P_L_500 = np.array(sorted(P_L_500))[:,1]
P_L_1000 = np.array(sorted(P_L_1000))[:,1]
P_L_1500 = np.array(sorted(P_L_1500))[:,1]
P_L_2000 = np.array(sorted(P_L_2000))[:,1]
#%%
sims = np.array([i**2 for i in range(5, 16)])

plt.figure(figsize=(10, 10))
#plt.plot(sims, A_L_500, 'b-x', label="All L=500")
#plt.plot(sims, A_L_1000, 'g-x', label="All L=1000")
#plt.plot(sims, A_L_1500, 'k-x', label="All L=1500")
#plt.plot(sims, A_L_2000, 'r-x', label="All L=2000")
plt.plot(sims, B_L_500, 'b-o', label="GPU L=500")
plt.plot(sims, B_L_1000, 'g-o', label="GPU L=1000")
plt.plot(sims, B_L_1500, 'k-o', label="GPU L=1500")
plt.plot(sims, B_L_2000, 'r-o', label="GPU L=2000")
plt.plot(sims, P_L_500, 'b-d', label="Python L=500")
plt.plot(sims, P_L_1000, 'g-d', label="Python L=1000")
plt.plot(sims, P_L_1500, 'k-d', label="Python L=1500")
plt.plot(sims, P_L_2000, 'r-d', label="Python L=2000")
plt.grid(True)
plt.xscale('log')
plt.yscale('log')
plt.xlabel('Number of simulations')
plt.ylabel('Time [s]')
plt.legend()
plt.show()


#%%
import numpy as np
import matplotlib.pyplot as plt

n_sims = np.array([(10*i)**2 for i in range(2, 10)])

cpu = np.array([
  367.07748079299927, 
  814.9095788002014,
  1456.0895714759827,
  2287.404059410095,
  2475.719622850418,
  4375.263347387314,
  5410.338870048523,
  5559.4789299964905
])
  
gpu = np.array([ 
  625.650000,
  1398.650000,
  2482.780000,
  3876.410000,
  5577.930000,
  7587.810000,
  9905.240000,
  12544.570000
])

plt.plot(n_sims[:len(gpu)], gpu, 'b-o', label="GPU")
plt.plot(n_sims[:len(cpu)], cpu, 'r-x', label="CPU")
plt.xscale('log')
plt.yscale('log')
plt.xlabel("Number of simulations")
plt.ylabel("Time [s]")
plt.grid(True)
plt.legend()
plt.show()

#%%
import numpy as np
import matplotlib.pyplot as plt

n_sims = np.array([(i)**2 for i in range(1, 10)])

cpu = np.array([
    0.6890361309051514,
    2.523871660232544,
    9.221802473068237,
    13.638726234436035,
    27.869741439819336,
    34.807886362075806,
    30.59798765182495,
    40.22039723396301,
    50.838963747024536
])
  
gpu = np.array([ 
    0.740000,
    1.370000,
    2.440000,
    3.920000,
    5.820000,
    8.130000,
    10.850000,
    14.050000,
    17.650000
])

plt.plot(n_sims[:len(gpu)], gpu, 'b-o', label="GPU")
plt.plot(n_sims[:len(cpu)], cpu, 'r-x', label="CPU")
plt.xscale('log')
plt.yscale('log')
plt.xlabel("Number of simulations")
plt.ylabel("Time [s]")
plt.grid(True)
plt.legend()
plt.show()



#%%
import os 
import matplotlib.pyplot as plt
import numpy as np

DIR_GPU = "/home/dsanmartin/Dropbox/UTFSM/Magister/Thesis/output_GPU/"
DIR_CPU = "/home/dsanmartin/Dropbox/UTFSM/Magister/Thesis/output_CPU/"

gpu_folders = [dI for dI in os.listdir(DIR_GPU) if os.path.isdir(os.path.join(DIR_GPU, dI))]
cpu_folders = [dI for dI in os.listdir(DIR_CPU) if os.path.isdir(os.path.join(DIR_CPU, dI))]


gpu_times = list()
cpu_times = list()


for sim_dir in gpu_folders:
  with open(DIR_GPU + sim_dir + "/log.txt") as f:
    lines = f.readlines()
    n_sims = int(lines[1].split()[-1])
    time = float(lines[25].split()[-2])
    gpu_times.append((n_sims, time))
    
      
for sim_dir in cpu_folders:
  with open(DIR_CPU + sim_dir + "/log.txt") as f:
    lines = f.readlines()
    n_sims = int(lines[1].split()[-1])
    time = float(lines[25].split()[-2])
    cpu_times.append((n_sims, time))
      
      
gpu_times = np.array(sorted(gpu_times))[:,1]
cpu_times = np.array(sorted(cpu_times))[:,1]
#%%
cpu_times = np.array([
    0.6498599052429199,
    2.580031394958496,
    5.675163745880127,
    10.04686450958252,
    16.167478799819946,
    22.706374168395996,
    30.90945816040039,
    56.80656957626343,
    51.20234990119934,
    63.094027042388916,
    248.88072729110718,
    577.7576701641083,
    1091.6537563800812,
    1564.9424171447754,
    2248.5730969905853,
    3066.139952659607,
    4011.6433086395264,
    5132.108956575394
])

#%%
sims = np.array([i**2 for i in range(1, 11)] + [(10*i)**2 for i in range(2, 10)])

plt.figure(figsize=(10, 10))
plt.plot(sims, cpu_times, 'b-o', label="CPU")
plt.plot(sims, gpu_times, 'g-d', label="GPU")
#plt.plot(sims, cpu_times/gpu_times, 'r-x', label="Speedup")
plt.grid(True)
plt.xscale('log')
plt.yscale('log')
plt.xlabel('Number of simulations')
plt.ylabel('Time [s]')
plt.legend()
plt.show()
