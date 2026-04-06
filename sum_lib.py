"""Funciones compartidas para el ejercicio de suma de vector en paralelo."""

from __future__ import annotations

import numpy as np
from fractions import Fraction
from typing import Iterable, List, Sequence

DEFAULT_SEED = 2026
rng = np.random.default_rng(DEFAULT_SEED)
DEFAULT_VECTOR_SIZE = 100_000_000
DEFAULT_VECTOR: np.ndarray = rng.random(DEFAULT_VECTOR_SIZE)


def sum_elements(v: np.ndarray) -> float:
    """Calcula la suma de los elementos de un vector con un loop Python puro."""
    res = 0
    for e in v:
        res += e
    return res


def mean_serial(values: Iterable[float]) -> float:
    """Calcula el promedio en forma secuencial."""
    total = 0.0
    count = 0
    for value in values:
        total += value
        count += 1

    if count == 0:
        raise ValueError("No se puede calcular promedio de una lista vacia")

    return total / count


def print_result(title: str, vector_size: int, workers: int, total_sum: float, elapsed: float) -> None:
    """Imprime un resumen simple para comparar variantes."""
    print(f"Implementacion: {title}")
    print(f"Tamanio del vector: {vector_size}")
    print(f"Workers: {workers}")
    print(f"Tiempo (segundos): {elapsed:.6f}")
    print(f"Suma total: {total_sum:.6f}")
