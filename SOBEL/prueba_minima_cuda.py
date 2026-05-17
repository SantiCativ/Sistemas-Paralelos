from numba import cuda
import numpy as np


@cuda.jit
def add_kernel(a, b, c):
    i = cuda.grid(1)

    if i < c.size:
        c[i] = a[i] + b[i]


n = 1000000

a = np.ones(n)
b = np.ones(n)
c = np.zeros(n)

threadsperblock = 256
blockspergrid = (n + threadsperblock - 1) // threadsperblock

add_kernel[blockspergrid, threadsperblock](a, b, c)

print(c[:10])
