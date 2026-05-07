# Sobel

## Scripts individuales

Ejecutar una imagen con 5 corridas y tiempos promedio:

```bash
python sobel_secuencial.py --r=5 --image=6000x6000
python sobel_numpy.py --r=5 --image=6000x6000
python sobel_numba_parallel.py --r=5 --image=6000x6000
```

Si se omite `--r`, el script hace una corrida, muestra los tiempos en terminal y abre la imagen resultante:

```bash
python sobel_secuencial.py --image=750x750
```

## Benchmark CSV

Por defecto recorre `750x750`, `1500x1500`, `3000x3000` y `6000x6000`, con 5 corridas por metodo:

```bash
python benchmark_sobel.py
```

Guardar el CSV en un archivo:

```bash
python benchmark_sobel.py --output=resultados_sobel.csv
```

Limitar la cantidad de scripts ejecutados en paralelo:

```bash
python benchmark_sobel.py --workers=4
```

El CSV incluye:

```text
method,image,runs,rgb_gris_promedio_s,sobel_promedio_s,total_promedio_s,blancos_pct,speed_up,performance_pct
```

`speed_up` se calcula como `tiempo_total_secuencial / tiempo_total_metodo`.
`performance_pct` se calcula como `speed_up * 100`.
