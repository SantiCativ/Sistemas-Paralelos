"""
Multiplicación de matrices - Threading.
Usa threads de Python para dividir las filas de C entre workers.
NOTA: limitado por el GIL de CPython en tareas CPU-bound.
"""
import argparse
import time
import numpy as np
import threading

def generate_matrix(n, seed):
    return np.random.default_rng(seed).random((n, n)).tolist()

def transpose(B, n):
    return [[B[j][i] for j in range(n)] for i in range(n)]

def worker_rows(A, Bt, C, rows, n):
    for i in rows:
        A_i = A[i]
        for j in range(n):
            s = 0.0
            Bt_j = Bt[j]
            for k in range(n):
                s += A_i[k] * Bt_j[k]
            C[i][j] = s

def matmul_threading(A, B, n, num_workers):
    Bt = transpose(B, n)
    C = [[0.0] * n for _ in range(n)]
    threads = []
    # Repartir filas entre workers
    chunks = [list(range(i, n, num_workers)) for i in range(num_workers)]
    for rows in chunks:
        t = threading.Thread(target=worker_rows, args=(A, Bt, C, rows, n))
        threads.append(t)
        t.start()
    for t in threads:
        t.join()
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
    C = matmul_threading(A, B, n, args.workers)
    elapsed = time.perf_counter() - start

    cs = checksum(C)
    print(f"method,workers,complexity,time,checksum")
    print(f"threading,{args.workers},{n},{elapsed:.6f},{cs:.6f}")

if __name__ == "__main__":
    main()
