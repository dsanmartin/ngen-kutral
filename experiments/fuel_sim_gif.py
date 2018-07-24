import numpy as np
import matplotlib.pyplot as plt

SIM_NAME = '20180719163918'
DIR_BASE = "/media/dsanmartin/My Passport/Data/Thesis/risk_map/" + SIM_NAME + "/"
SAVE_DIR = "/home/dsanmartin/Desktop/GIF_random/"
#SIM_NAME = "test2/"
#DIR_BASE = "/home/dsanmartin/Desktop/" + SIM_NAME + "/"

row = 11
col = 11

for i in range(row):
  for j in range(col):
    #print(i, j)
    BB = np.load(DIR_BASE + 'B_' + str(i) + '-' + str(j) +  '.npy')
    
    k = row * i + j
    
    if k < 10:
      name = '00' + str(k) + '.png'
    elif k < 100:
      name = '0' + str(k) + '.png'
    else:
      name = '' + str(k) + '.png'
    #plt.show()
    plt.imshow(BB[-1], origin="lower")
    plt.savefig(SAVE_DIR + name)
#%%
