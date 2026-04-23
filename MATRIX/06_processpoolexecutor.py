"""Multiplicación de matrices — concurrent.futures.ProcessPoolExecutor."""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
from time import perf_counter

from matrix_lib import (
    generate_matrix, transpose, matmul_rows, checksum,
    print_result, DEFAULT_COMPLEXITY, DEFAULT_SEED
)


def compute_chunk(args):
    A, Bt, rows, n = args
    return matmul_rows(A, Bt, rows, n)


def main() -> None:
    parser = argparse.ArgumentParser(description="Multiplicación de matrices — ProcessPoolExecutor")
    parser.add_argument("--complexity", type=int, default=DEFAULT_COMPLEXITY)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()

    n = args.complexity
    A = generate_matrix(n, args.seed)
    B = generate_matrix(n, args.seed + 1)
    Bt = transpose(B, n)

    chunks = [list(range(i, n, args.workers)) for i in range(args.workers)]
    tasks = [(A, Bt, chunk, n) for chunk in chunks]

    C = [[0.0] * n for _ in range(n)]

    start = perf_counter()
    with ProcessPoolExecutor(max_workers=args.workers) as executor:
        results = list(executor.map(compute_chunk, tasks))
    elapsed = perf_counter() - start

    for chunk_result in results:
        for i, row in chunk_result:
            C[i] = row

    print_result("processpoolexecutor", args.workers, n, checksum(C), elapsed)


if __name__ == "__main__":
    main()
