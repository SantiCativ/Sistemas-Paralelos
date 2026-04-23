"""Multiplicación de matrices — Secuencial tradicional (triple loop, acceso por columna)."""

from __future__ import annotations

import argparse
from time import perf_counter

from matrix_lib import generate_matrix, transpose, checksum, print_result, DEFAULT_COMPLEXITY, DEFAULT_SEED


def matmul_traditional(A, B, n):
    C = [[0.0] * n for _ in range(n)]
    for i in range(n):
        for j in range(n):
            s = 0.0
            for k in range(n):
                s += A[i][k] * B[k][j]
            C[i][j] = s
    return C


def main() -> None:
    parser = argparse.ArgumentParser(description="Multiplicación de matrices — Secuencial tradicional")
    parser.add_argument("--complexity", type=int, default=DEFAULT_COMPLEXITY)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--workers", type=int, default=1)  # ignorado
    args = parser.parse_args()

    n = args.complexity
    A = generate_matrix(n, args.seed)
    B = generate_matrix(n, args.seed + 1)

    start = perf_counter()
    C = matmul_traditional(A, B, n)
    elapsed = perf_counter() - start

    print_result("sequential_traditional", 1, n, checksum(C), elapsed)


if __name__ == "__main__":
    main()
