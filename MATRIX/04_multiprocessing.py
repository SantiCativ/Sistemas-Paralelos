"""
Multiplicación de matrices - Multiprocessing.
Divide filas entre procesos reales (sin limitación del GIL).
"""
import argparse
import time
import numpy as np
import multiprocessing as mp

def generate_matrix(n, seed):
    return np.random.default_rng(seed).random((n, n)).tolist()

def transpose(B, n):
    return [[B[j][i] for j in range(n)] for i in range(n)]

def compute_rows(args):
    A, Bt, rows, n = args
    result = []
    for i in rows:
        A_i = A[i]
        row_result = []
        for j in range(n):
            s = 0.0
            Bt_j = Bt[j]
            for k in range(n):
                s += A_i[k] * Bt_j[k]
            row_result.append(s)
        result.append((i, row_result))
    return result

def matmul_multiprocessing(A, B, n, num_workers):
    Bt = transpose(B, n)
    C = [[0.0] * n for _ in range(n)]
    chunks = [list(range(i, n, num_workers)) for i in range(num_workers)]
    tasks = [(A, Bt, chunk, n) for chunk in chunks]

    with mp.Pool(processes=num_workers) as pool:
        results = pool.map(compute_rows, tasks)

    for chunk_result in results:
        for i, row in chunk_result:
            C[i] = row
    return C

def checksum(C):
    return sum(C[i][j] for i in range(len(C)) for j in range(len(C[0])))

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--complexity", type=int, default=512)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()

    n = args.complexity
    A = generate_matrix(n, args.seed)
    B = generate_matrix(n, args.seed + 1)

    start = time.perf_counter()
    C = matmul_multiprocessing(A, B, n, args.workers)
    elapsed = time.perf_counter() - start

    cs = checksum(C)
    print(f"method,workers,complexity,time,checksum")
    print(f"multiprocessing,{args.workers},{n},{elapsed:.6f},{cs:.6f}")

if __name__ == "__main__":
    main()
