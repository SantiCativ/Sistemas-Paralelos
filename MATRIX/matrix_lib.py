"""Funciones compartidas para el ejercicio de multiplicación de matrices en paralelo."""

from __future__ import annotations

import numpy as np
from typing import List

DEFAULT_SEED = 2026
DEFAULT_COMPLEXITY = 512


def generate_matrix(n: int, seed: int) -> List[List[float]]:
    """Genera una matriz NxN de floats aleatorios usando numpy.random.default_rng."""
    return np.random.default_rng(seed).random((n, n)).tolist()


def transpose(B: List[List[float]], n: int) -> List[List[float]]:
    """Retorna la transpuesta de una matriz NxN."""
    return [[B[j][i] for j in range(n)] for i in range(n)]


def matmul_rows(A: List[List[float]], Bt: List[List[float]], rows: List[int], n: int) -> List[tuple]:
    """
    Multiplica las filas indicadas de A por Bt (B transpuesta).
    Retorna lista de (índice_fila, resultado_fila).
    Usada por todos los métodos paralelos como unidad de trabajo.
    """
    result = []
    for i in rows:
        A_i = A[i]
        row = []
        for j in range(n):
            s = 0.0
            Bt_j = Bt[j]
            for k in range(n):
                s += A_i[k] * Bt_j[k]
            row.append(s)
        result.append((i, row))
    return result


def checksum(C: List[List[float]]) -> float:
    """Suma todos los elementos de la matriz resultado."""
    return sum(C[i][j] for i in range(len(C)) for j in range(len(C[0])))


def print_result(title: str, workers: int, complexity: int, total_checksum: float, elapsed: float) -> None:
    """Imprime en formato CSV: method,workers,complexity,time,checksum."""
    print(f"method,workers,complexity,time,checksum")
    print(f"{title},{workers},{complexity},{elapsed:.6f},{total_checksum:.6f}")
