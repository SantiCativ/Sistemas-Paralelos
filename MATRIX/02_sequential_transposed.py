"""Multiplicación de matrices — Secuencial transpuesta (mejora localidad de caché)."""

from __future__ import annotations

import argparse
from time import perf_counter

from matrix_lib import generate_matrix, transpose, checksum, print_result, DEFAULT_COMPLEXITY, DEFAULT_SEED


def matmul_transposed(A, Bt, n):
    C = [[0.0] * n for _ in range(n)]
    for i in range(n):
        A_i = A[i]
        for j in range(n):
            s = 0.0
            Bt_j = Bt[j]
            for k in range(n):
                s += A_i[k] * Bt_j[k]
            C[i][j] = s
    return C


def main() -> None:
    parser = argparse.ArgumentParser(description="Multiplicación de matrices — Secuencial transpuesta")
    parser.add_argument("--complexity", type=int, default=DEFAULT_COMPLEXITY)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--workers", type=int, default=1)  # ignorado
    args = parser.parse_args()

    n = args.complexity
    A = generate_matrix(n, args.seed)
    B = generate_matrix(n, args.seed + 1)
    Bt = transpose(B, n)

    start = perf_counter()
    C = matmul_transposed(A, Bt, n)
    elapsed = perf_counter() - start

    print_result("sequential_transposed", 1, n, checksum(C), elapsed)


if __name__ == "__main__":
    main()
