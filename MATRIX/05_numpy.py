"""
Multiplicación de matrices - NumPy.
Usa np.dot / @ que internamente llama a BLAS/LAPACK optimizados.
"""
import argparse
import time
import numpy as np

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--complexity", type=int, default=512)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--workers", type=int, default=1)
    args = parser.parse_args()

    n = args.complexity
    rng = np.random.default_rng(args.seed)
    A = rng.random((n, n))
    rng2 = np.random.default_rng(args.seed + 1)
    B = rng2.random((n, n))

    start = time.perf_counter()
    C = A @ B
    elapsed = time.perf_counter() - start

    cs = float(C.sum())
    print(f"method,workers,complexity,time,checksum")
    print(f"numpy,1,{n},{elapsed:.6f},{cs:.6f}")

if __name__ == "__main__":
    main()
