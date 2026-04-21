"""
Multiplicación de matrices - Numba @njit con parallel=True.
Usa prange para paralelizar el loop externo automáticamente
sobre todos los núcleos disponibles (vía OpenMP).

Este script es un EXTRA al trabajo práctico — no reemplaza a 06_numba.py.
Su objetivo es comparar el beneficio adicional del paralelismo de Numba
sobre la compilación JIT sola (parallel=False).
"""
import argparse
import time
import numpy as np

try:
    from numba import njit, prange
    NUMBA_AVAILABLE = True
except ImportError:
    NUMBA_AVAILABLE = False


if NUMBA_AVAILABLE:
    @njit(parallel=True, cache=True)
    def matmul_numba_parallel(A, B):
        n = A.shape[0]
        C = np.zeros((n, n), dtype=np.float64)
        for i in prange(n):       # prange → paralelizado automáticamente
            for k in range(n):
                for j in range(n):
                    C[i, j] += A[i, k] * B[k, j]
        return C


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--complexity", type=int, default=512)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--workers", type=int, default=1)  # ignorado, Numba usa todos los núcleos
    args = parser.parse_args()

    n = args.complexity

    print("method,workers,complexity,time,checksum")

    if not NUMBA_AVAILABLE:
        print(f"numba_njit_parallel,1,{n},ERROR_NOT_INSTALLED,0")
        return

    rng = np.random.default_rng(args.seed)
    A = rng.random((n, n))
    rng2 = np.random.default_rng(args.seed + 1)
    B = rng2.random((n, n))

    # Warmup — compilación JIT (no se mide)
    print("# Warmup numba parallel (compilación JIT)...", flush=True)
    _ = matmul_numba_parallel(A[:4, :4], B[:4, :4])

    # Medición real
    start = time.perf_counter()
    C = matmul_numba_parallel(A, B)
    elapsed = time.perf_counter() - start

    cs = float(C.sum())
    print(f"numba_njit_parallel,1,{n},{elapsed:.6f},{cs:.6f}")
    print(f"# Núcleos usados: automático (todos los disponibles vía OpenMP)")


if __name__ == "__main__":
    main()
