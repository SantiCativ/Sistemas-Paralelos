from __future__ import annotations

import argparse
import threading
from queue import Queue
from time import time

import numpy as np

from sum_lib import DEFAULT_VECTOR, DEFAULT_VECTOR_SIZE, sum_elements, print_result


def worker(task_queue: Queue, results: list[float]) -> None:
    while True:
        item = task_queue.get()
        if item is None:
            break

        index, chunk = item
        results[index] = sum_elements(chunk)
        task_queue.task_done()


def main() -> None:
    parser = argparse.ArgumentParser(description="Suma de vector en paralelo con threading")
    parser.add_argument("--size", type=int, default=None, help="Tamanio del vector. Ejemplo: 1000000")
    parser.add_argument("--workers", type=int, default=4, help="Cantidad de hilos")
    args = parser.parse_args()

    if args.size is None:
        vector = DEFAULT_VECTOR
    else:
        rng = np.random.default_rng(2026)
        vector = rng.random(args.size)

    # Dividir el vector en chunks iguales para cada worker
    chunks = np.array_split(vector, args.workers)

    task_queue: Queue = Queue()
    results: list[float] = [0.0] * args.workers

    threads: list[threading.Thread] = []
    for _ in range(args.workers):
        thread = threading.Thread(target=worker, args=(task_queue, results))
        thread.start()
        threads.append(thread)

    init = time()
    for index, chunk in enumerate(chunks):
        task_queue.put((index, chunk))

    task_queue.join()

    for _ in threads:
        task_queue.put(None)

    for thread in threads:
        thread.join()

    result = sum(results)
    end = time() - init

    print_result("threading", len(vector), args.workers, result, end)
    print(f"With C =", len(vector))


if __name__ == "__main__":
    main()
