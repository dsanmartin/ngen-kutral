import matplotlib.pyplot as plt
import numpy as np

dt_dir = '1e-8'
L = 500

# CONVERGENCE TEST
exp_dir = str(L) + "/" + dt_dir + "/"

U_4096 = np.load('convergence/' + exp_dir + 'U_4096.npy')
U_2048 = np.load('convergence/' + exp_dir + 'U_2048.npy')
U_1024 = np.load('convergence/' + exp_dir + 'U_1024.npy')
U_512 = np.load('convergence/' + exp_dir + 'U_512.npy')
U_256 = np.load('convergence/' + exp_dir + 'U_256.npy')
U_128 = np.load('convergence/' + exp_dir + 'U_128.npy')
U_64 = np.load('convergence/' + exp_dir + 'U_64.npy')

B_4096 = np.load('convergence/' + exp_dir + 'B_4096.npy')
B_2048 = np.load('convergence/' + exp_dir + 'B_2048.npy')
B_1024 = np.load('convergence/' + exp_dir + 'B_1024.npy')
B_512 = np.load('convergence/' + exp_dir + 'B_512.npy')
B_256 = np.load('convergence/' + exp_dir + 'B_256.npy')
B_128 = np.load('convergence/' + exp_dir + 'B_128.npy')
B_64 = np.load('convergence/' + exp_dir + 'B_64.npy')


# Reference 1024
#errors_u = np.array([
#    np.linalg.norm((U_64 - U_1024[::16, ::16]).flatten(), np.inf),
#    np.linalg.norm((U_128 - U_1024[::8, ::8]).flatten(), np.inf),
#    np.linalg.norm((U_256 - U_1024[::4, ::4]).flatten(), np.inf),
#    np.linalg.norm((U_512 - U_1024[::2, ::2]).flatten(), np.inf),
#    ])
#
#errors_b = np.array([
#    np.linalg.norm((B_64 - B_1024[::16, ::16]).flatten(), np.inf),
#    np.linalg.norm((B_128 - B_1024[::8, ::8]).flatten(), np.inf),
#    np.linalg.norm((B_256 - B_1024[::4, ::4]).flatten(), np.inf),
#    np.linalg.norm((B_512 - B_1024[::2, ::2]).flatten(), np.inf),
#    ])

# Reference 2048
# errors_u = np.array([
#     np.linalg.norm((U_64 - U_2048[::32, ::32]).flatten(), np.inf),
#     np.linalg.norm((U_128 - U_2048[::16, ::16]).flatten(), np.inf),
#     np.linalg.norm((U_256 - U_2048[::8, ::8]).flatten(), np.inf),
#     np.linalg.norm((U_512 - U_2048[::4, ::4]).flatten(), np.inf),
#     np.linalg.norm((U_1024 - U_2048[::2, ::2]).flatten(), np.inf),
#     ])

# errors_b = np.array([
#     np.linalg.norm((B_64 - B_2048[::32, ::32]).flatten(), np.inf),
#     np.linalg.norm((B_128 - B_2048[::16, ::16]).flatten(), np.inf),
#     np.linalg.norm((B_256 - B_2048[::8, ::8]).flatten(), np.inf),
#     np.linalg.norm((B_512 - B_2048[::4, ::4]).flatten(), np.inf),
#     np.linalg.norm((B_1024 - B_2048[::2, ::2]).flatten(), np.inf),
#     ])

# Reference 4096
errors_u = np.array([
    #np.linalg.norm((U_64 - U_4096[::64, ::64]).flatten(), np.inf),
    #np.linalg.norm((U_128 - U_4096[::32, ::32]).flatten(), np.inf),
    np.linalg.norm((U_256 - U_4096[::16, ::16]).flatten(), np.inf),
    np.linalg.norm((U_512 - U_4096[::8, ::8]).flatten(), np.inf),
    np.linalg.norm((U_1024 - U_4096[::4, ::4]).flatten(), np.inf),
    np.linalg.norm((U_2048 - U_4096[::2, ::2]).flatten(), np.inf)
    ])


#h = np.array([90/(2**i) for i in range(6, 12)])
h = np.array([90/(2**i) for i in range(8, 12)])

print(h)
print(errors_u)


plt.plot(h, errors_u, 'b-x')
plt.plot(h, h)
plt.plot(h, h**2)
plt.xlabel("h")
plt.grid(True)
plt.xscale('log')
plt.yscale('log')
plt.show()
