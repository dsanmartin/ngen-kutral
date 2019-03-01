import numpy as np
import matplotlib.pyplot as plt
DIR_BASE = "../ngen-kutral-gpu/test/output/201902261510170/" #GPU block 256 DB
#DIR_BASE = "/home/dsanmartin/Desktop/output_block/20190223195304/" GPU block 256 DB
#DIR_BASE = "/home/dsanmartin/Desktop/output/20190224004754/" Python
#DIR_BASE = "/home/dsanmartin/Desktop/cluster_backups/output_all_512DB/20190224172918/"
DIR_BASE = "/home/dsanmartin/Desktop/201902261745360/"
sim = '55'
U00 = np.loadtxt(DIR_BASE + "U0_" + sim + ".txt")
B00 = np.loadtxt(DIR_BASE + "B0_" + sim + ".txt")
U0 = U00.reshape((128, 128))
B0 = B00.reshape((128, 128))
Uaa = np.loadtxt(DIR_BASE + "U_" + sim + ".txt")
Baa = np.loadtxt(DIR_BASE + "B_" + sim + ".txt")
Ua = Uaa.reshape((128, 128))
Ba = Baa.reshape((128, 128))
plt.figure(figsize=(8, 6))
plt.subplot(2, 2, 1)
plt.imshow(U0, origin="lower", extent=[0, 90, 0, 90], cmap=plt.cm.jet)
plt.colorbar()
#plt.show()
plt.subplot(2, 2, 2)
plt.imshow(B0, origin="lower", extent=[0, 90, 0, 90], cmap=plt.cm.Oranges)
plt.colorbar()
#plt.show()
plt.subplot(2, 2, 3)
plt.imshow(Ua, origin="lower", extent=[0, 90, 0, 90], cmap=plt.cm.jet)
plt.colorbar()
#plt.show()
plt.subplot(2, 2, 4)
plt.imshow(Ba, origin="lower",  extent=[0, 90, 0, 90], cmap=plt.cm.Oranges)
plt.colorbar()
plt.tight_layout()
plt.show()
print(np.all(U0 == Ua) and np.all(B0 == Ba))
#%%
n_sim = 15
sim_size = 2*128*128
block_size = 256
size = n_sim*2*128*128
grid = (size + block_size - 1) // block_size
for bl in range(grid):
  for tr in range(block_size):
  
    block = bl#grid - 1
    thread = tr#255
    tid = block*block_size+thread
    sim = tid // (sim_size)
    off = sim*2*128*128
    i = (tid - off) % (2*128)
    j = (tid - off) // (2*128)
    #i = (tid - off) % (128*128)
    #j = (tid - off) // (128*128)
    upos = off + 128 * j + i
    bpos = off + 128 * j + i + 128*128
    
    if i > 127 or j > 127:
      print("simulations size: ", size)
      print("one simulation size: ", sim_size)
      print("tId: ", tid)
      print("sim ", sim)
      print("off: ", off)
      print("i: ", i)
      print("j: ", j)
      print("RHS lim: ", 2*sim*128*128, ", ", (2*sim+1)*128*128)
      print("upos: ", upos)
      print("bpos: ", bpos, "-: ", bpos-128*128)
      print("dentro dominio: ", not(i == 0 or i == 127 or j == 0 or j == 127))

#%%
y_size = 2
x_size = 2
for i in range(y_size):
  for j in range(x_size):
    print(j*y_size + i*x_size)

#%%
n_sim = 4
sim_size = 128*128
block_size = 256
size = n_sim*sim_size
grid = (size + block_size - 1) // block_size
#for bl in range(grid):
#  for tr in range(block_size):
  
block = 0#grid - 1
thread = 255#255
tid = block*block_size+thread
sim = tid // (sim_size)
off = sim*128*128
i = (tid) % 128
j = (tid) // 128
print((tid - off) % 128)
print((tid - off) // 128)
#i = (tid - off) % 128
#j = (tid - off) // 128
#i = (tid - off) % (128*128)
#j = (tid - off) // (128*128)
gindex = off + j * 128 + i
upos = gindex
bpos = gindex + 128*128

if True:#i > 127 or j > 127:
  print("simulations size: ", size)
  print("one simulation size: ", sim_size)
  print("tId: ", tid)
  print("sim ", sim)
  print("off: ", off)
  print("i: ", i)
  print("j: ", j)
  print("RHS lim: ", 2*sim*128*128, ", ", (2*sim+1)*128*128)
  print("upos: ", upos)
  print("bpos: ", bpos, "-: ", bpos-128*128)
  print("dentro dominio: ", not(i == 0 or i == 127 or j == 0 or j == 127))