"""Multiplicación de matrices — Numba @njit parallel=False (compilación JIT sin paralelismo)."""

from __future__ import annotations

import argparse
from time import perf_counter

import numpy as np

from matrix_lib import print_result, DEFAULT_COMPLEXITY, DEFAULT_SEED

try:
    from numba import njit
    NUMBA_AVAILABLE = True
except ImportError:
    NUMBA_AVAILABLE = False

if NUMBA_AVAILABLE:
    @njit(cache=True)
    def matmul_numba(A, B):
        n = A.shape[0]
        C = np.zeros((n, n), dtype=np.float64)
        for i in range(n):
            for k in range(n):
                for j in range(n):
                    C[i, j] += A[i, k] * B[k, j]
        return C


def main() -> None:
    parser = argparse.ArgumentParser(description="Multiplicación de matrices — Numba njit (parallel=False)")
    parser.add_argument("--complexity", type=int, default=DEFAULT_COMPLEXITY)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--workers", type=int, default=1)  # ignorado
    args = parser.parse_args()

    n = args.complexity

    if not NUMBA_AVAILABLE:
        print("method,workers,complexity,time,checksum")
        print(f"numba_njit,1,{n},ERROR_NOT_INSTALLED,0")
        return

    A = np.random.default_rng(args.seed).random((n, n))
    B = np.random.default_rng(args.seed + 1).random((n, n))

    # Warmup — compilación JIT (no se mide)
    print("# Warmup numba njit (compilación JIT)...", flush=True)
    _ = matmul_numba(A[:2, :2], B[:2, :2])

    start = perf_counter()
    C = matmul_numba(A, B)
    elapsed = perf_counter() - start

    print_result("numba_njit", 1, n, float(C.sum()), elapsed)


if __name__ == "__main__":
    main()
