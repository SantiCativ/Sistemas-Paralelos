from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
from time import time

import numpy as np

from sum_lib import DEFAULT_VECTOR, DEFAULT_VECTOR_SIZE, sum_elements, print_result


def sum_chunk(chunk: np.ndarray) -> float:
    """Suma los elementos de un fragmento del vector."""
    return sum_elements(chunk)


def main() -> None:
    parser = argparse.ArgumentParser(description="Suma de vector en paralelo con ThreadPoolExecutor")
    parser.add_argument("--size", type=int, default=None, help="Tamanio del vector. Ejemplo: 1000000")
    parser.add_argument("--workers", type=int, default=4, help="Cantidad de workers")
    args = parser.parse_args()

    if args.size is None:
        vector = DEFAULT_VECTOR
    else:
        rng = np.random.default_rng(2026)
        vector = rng.random(args.size)

    # Dividir el vector en chunks iguales para cada worker
    chunks = np.array_split(vector, args.workers)

    init = time()
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        partial_sums = list(executor.map(sum_chunk, chunks))
    result = sum(partial_sums)
    end = time() - init

    print_result("concurrent.futures.ThreadPoolExecutor", len(vector), args.workers, result, end)
    print(f"With C =", len(vector))


if __name__ == "__main__":
    main()
