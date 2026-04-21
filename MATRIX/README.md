# Benchmark — Multiplicación de Matrices
## Sistemas Paralelos — UNTDF 2026

### Estructura del proyecto

```
matrix_benchmark/
├── 01_sequential.py            ← Secuencial tradicional
├── 02_sequential_transposed.py ← Secuencial transpuesta
├── 03_threading.py             ← Threading (limitado por GIL)
├── 04_multiprocessing.py       ← Multiprocessing (procesos reales)
├── 05_numpy.py                 ← NumPy (BLAS optimizado)
├── 06_numba.py                 ← Numba @njit (compilación JIT)
├── 07_pytorch.py               ← PyTorch (CPU / CUDA / MPS)
├── run_all_benchmarks.py       ← Script maestro ← CORRÉ ESTE
└── README.md                   ← Este archivo
```

---

### Requisitos

```bash
pip install numpy numba torch
# (numba y torch son opcionales; el runner los detecta automáticamente)
```
---

---

### Activar y usar entorno

```bash
python -m venv venv
source venv/bin/activate
```
---




### Cómo ejecutar

#### Opción A — Ejecución completa (recomendada)
```bash
cd matrix_benchmark
python run_all_benchmarks.py
```
Genera `resultados.csv` con todos los tiempos, checksums, speed-ups y eficiencias.

#### Opción B — Sin los métodos más lentos (complexity=1024 secuencial)
```bash
python run_all_benchmarks.py --skip-slow
```

#### Opción C — Script individual (para probar)
```bash
python 05_numpy.py --complexity 512 --seed 2026
```

---

### Parámetros del runner

| Parámetro | Default | Descripción |
|-----------|---------|-------------|
| `--output` | `resultados.csv` | Nombre del archivo CSV de salida |
| `--seed` | `2026` | Semilla para reproducibilidad |
| `--skip-slow` | off | Salta secuencial con N=1024 (puede tardar 30+ min) |

---

### Combinaciones ejecutadas

| Workers | Complexity | Scripts |
|---------|-----------|---------|
| 1       | 512       | Todos (base de referencia) |
| 4       | 512       | Threading + Multiprocessing |
| 4       | 1024      | Threading + Multiprocessing |
| N físicos | 1024    | Threading + Multiprocessing |

> Los métodos de un solo worker (numpy, numba, pytorch, secuenciales) se ejecutan solo una vez por complexity.

---

### Nota sobre checksums
El checksum debe ser idéntico entre todos los métodos para una misma `--complexity`.
Si hay diferencias, hay un error en la implementación.

⚠️ **Numba y NumPy pueden diferir levemente en el último decimal** por el orden de operaciones en punto flotante — es normal y esperado.
