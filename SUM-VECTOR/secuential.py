from __future__ import annotations

import argparse
from time import time

import numpy as np

from sum_lib import DEFAULT_VECTOR, DEFAULT_VECTOR_SIZE, sum_elements, print_result


def main() -> None:
    parser = argparse.ArgumentParser(description="Suma de vector en forma secuencial")
    parser.add_argument("--size", type=int, default=None, help="Tamanio del vector. Ejemplo: 1000000")
    parser.add_argument("--workers", type=int, default=1, help="Parametro informativo para mantener misma interfaz")
    args = parser.parse_args()

    if args.size is None:
        vector = DEFAULT_VECTOR
    else:
        rng = np.random.default_rng(2026)
        vector = rng.random(args.size)

    init = time()
    result = sum_elements(vector)
    end = time() - init

    print_result("secuencial", len(vector), args.workers, result, end)
    print(f"With C =", len(vector))


if __name__ == "__main__":
    main()
