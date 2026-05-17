from numba import cuda

print(cuda.is_available())
print(cuda.gpus)
