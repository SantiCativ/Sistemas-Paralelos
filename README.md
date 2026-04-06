# Ejercicios de Suma de Vector en Paralelo

Este directorio contiene versiones de un mismo ejercicio para comparar ejecucion secuencial y paralela en Python.

Objetivo del ejercicio:
- Calcular la suma de los elementos de un vector usando un loop Python puro.
- Dividir el vector en chunks para procesarlos en paralelo.
- Combinar las sumas parciales de forma secuencial al final.
- Comparar tiempos entre distintas implementaciones.

## Archivos

- `secuential.py`: version secuencial (baseline).
- `sum_threadpoolexecutor.py`: paralelo con `ThreadPoolExecutor`.
- `sum_processpoolexecutor.py`: paralelo con `ProcessPoolExecutor`.
- `sum_multiprocessing.py`: paralelo con `multiprocessing.Pool`.
- `sum_threading.py`: paralelo con `threading` manual.
- `sum_lib.py`: funciones compartidas.

## Requisitos

```bash
pip install numpy
```

Python 3.10+ recomendado.

## Como ejecutar

Ejecutar baseline secuencial:

```bash
python secuential.py --workers 1
```

Ejecutar variantes paralelas:

```bash
python sum_threadpoolexecutor.py --workers 4
python sum_processpoolexecutor.py --workers 4
python sum_multiprocessing.py --workers 4
python sum_threading.py --workers 4
```

## Entrada personalizada (opcional)

Todos los scripts aceptan `--size` para definir el tamanio del vector:

```bash
python secuential.py --size 10000000 --workers 1
python sum_processpoolexecutor.py --size 10000000 --workers 4
```

Si no se envia `--size`, se usa un vector por defecto de 100.000.000 elementos reproducible definido en `sum_lib.py`.

## Estrategia de paralelismo

El vector se divide en N chunks (uno por worker). Cada worker calcula la suma parcial de su chunk usando `sum_elements()` (loop Python puro). Al final, las sumas parciales se combinan de forma secuencial con `sum()`.

```
Vector completo:  [e0, e1, ..., eN]
                        |
              +---------+---------+
              |         |         |
           Chunk 0   Chunk 1   Chunk 2    <-- workers en paralelo
           sum(...)  sum(...)  sum(...)
              |         |         |
              +---------+---------+
                        |
                   sum(parciales)          <-- secuencial
```

## Que comparar en clase

- `Tiempo (segundos)` entre variantes.
- `Suma total` para verificar que todas den el mismo resultado.
- Diferencias entre hilos y procesos en una tarea CPU-bound.

## Pregunta de reflexion

Explicar por que, para este tipo de trabajo CPU-bound en CPython, el uso de procesos suele escalar mejor que el uso de hilos. (Pista: GIL — Global Interpreter Lock.)
