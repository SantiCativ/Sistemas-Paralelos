"""Multiplicación de matrices — threading.Thread con Queue (mismo patrón que sum_threading.py)."""

from __future__ import annotations

import argparse
import threading
from queue import Queue
from time import perf_counter

from matrix_lib import (
    generate_matrix, transpose, matmul_rows, checksum,
    print_result, DEFAULT_COMPLEXITY, DEFAULT_SEED
)


def worker(task_queue: Queue, C: list) -> None:
    while True:
        item = task_queue.get()
        if item is None:
            break
        A, Bt, rows, n = item
        row_results = matmul_rows(A, Bt, rows, n)
        for i, row in row_results:
            C[i] = row
        task_queue.task_done()


def main() -> None:
    parser = argparse.ArgumentParser(description="Multiplicación de matrices — threading.Thread")
    parser.add_argument("--complexity", type=int, default=DEFAULT_COMPLEXITY)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()

    n = args.complexity
    A = generate_matrix(n, args.seed)
    B = generate_matrix(n, args.seed + 1)
    Bt = transpose(B, n)

    # Dividir filas en chunks por worker
    chunks = [list(range(i, n, args.workers)) for i in range(args.workers)]
    C = [[0.0] * n for _ in range(n)]

    task_queue: Queue = Queue()
    threads = []
    for _ in range(args.workers):
        t = threading.Thread(target=worker, args=(task_queue, C))
        t.start()
        threads.append(t)

    start = perf_counter()
    for chunk in chunks:
        task_queue.put((A, Bt, chunk, n))

    task_queue.join()

    for _ in threads:
        task_queue.put(None)
    for t in threads:
        t.join()

    elapsed = perf_counter() - start

    print_result("threading", args.workers, n, checksum(C), elapsed)


if __name__ == "__main__":
    main()
