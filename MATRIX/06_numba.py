"""
Multiplicación de matrices - Numba JIT.
Compila el loop triple a código nativo con @njit (parallel=False).
El primer llamado incluye tiempo de compilación JIT (warmup),
el segundo llamado mide solo la ejecución real.
"""
import argparse
import time
import numpy as np

try:
    from numba import njit

    @njit(cache=True)
    def matmul_numba(A, B):
        n = A.shape[0]
        C = np.zeros((n, n), dtype=np.float64)
        for i in range(n):
            for k in range(n):
                for j in range(n):
                    C[i, j] += A[i, k] * B[k, j]
        return C

    NUMBA_AVAILABLE = True
except ImportError:
    NUMBA_AVAILABLE = False

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--complexity", type=int, default=512)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--workers", type=int, default=1)
    args = parser.parse_args()

    n = args.complexity

    if not NUMBA_AVAILABLE:
        print(f"method,workers,complexity,time,checksum")
        print(f"numba_njit,1,{n},ERROR_NOT_INSTALLED,0")
        return

    rng = np.random.default_rng(args.seed)
    A = rng.random((n, n))
    rng2 = np.random.default_rng(args.seed + 1)
    B = rng2.random((n, n))

    # Warmup - compilación JIT (no se mide)
    print("# Warmup numba (compilación JIT)...", flush=True)
    _ = matmul_numba(A[:2, :2], B[:2, :2])

    # Medición real
    start = time.perf_counter()
    C = matmul_numba(A, B)
    elapsed = time.perf_counter() - start

    cs = float(C.sum())
    print(f"method,workers,complexity,time,checksum")
    print(f"numba_njit,1,{n},{elapsed:.6f},{cs:.6f}")

if __name__ == "__main__":
    main()
