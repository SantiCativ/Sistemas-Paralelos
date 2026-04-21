"""
Multiplicación de matrices - Método secuencial transpuesta.
Se transpone B para mejorar la localidad de caché (acceso por fila en vez de columna).
"""
import argparse
import time
import numpy as np

def generate_matrix(n, seed):
    return np.random.default_rng(seed).random((n, n)).tolist()

def transpose(B, n):
    return [[B[j][i] for j in range(n)] for i in range(n)]

def matmul_transposed(A, B, n):
    Bt = transpose(B, n)
    C = [[0.0] * n for _ in range(n)]
    for i in range(n):
        for j in range(n):
            s = 0.0
            Bt_j = Bt[j]
            A_i = A[i]
            for k in range(n):
                s += A_i[k] * Bt_j[k]
            C[i][j] = s
    return C

def checksum(C):
    return sum(C[i][j] for i in range(len(C)) for j in range(len(C[0])))

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--complexity", type=int, default=512)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--workers", type=int, default=1)
    args = parser.parse_args()

    n = args.complexity
    A = generate_matrix(n, args.seed)
    B = generate_matrix(n, args.seed + 1)

    start = time.perf_counter()
    C = matmul_transposed(A, B, n)
    elapsed = time.perf_counter() - start

    cs = checksum(C)
    print(f"method,workers,complexity,time,checksum")
    print(f"sequential_transposed,1,{n},{elapsed:.6f},{cs:.6f}")

if __name__ == "__main__":
    main()
