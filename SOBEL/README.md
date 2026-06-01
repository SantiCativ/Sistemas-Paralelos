# Sobel

## Scripts individuales

Ejecutar una imagen con 5 corridas y tiempos promedio:

```bash
python sobel_secuencial.py --r=5 --image=6000x6000
python sobel_numpy.py --r=5 --image=6000x6000
python sobel_numba_parallel.py --r=5 --image=6000x6000
python sobel_numba_cuda.py --r=5 --image=6000x6000
python sobel_pytorch.py --r=5 --image=6000x6000
```

Para PyTorch se puede elegir el dispositivo:

```bash
python sobel_pytorch.py --r=5 --image=6000x6000 --device=cpu
python sobel_pytorch.py --r=5 --image=6000x6000 --device=cuda
python sobel_pytorch.py --r=5 --image=6000x6000 --device=mps
```

Si se omite `--device`, PyTorch elige CUDA, luego MPS y finalmente CPU segun
disponibilidad. Si se omite `--r`, el script hace una corrida, muestra los
tiempos en terminal y abre la imagen resultante.

## Benchmark CSV

Por defecto recorre `750x750`, `1500x1500`, `3000x3000` y `6000x6000`, con 5
corridas por metodo. Incluye secuencial, NumPy, Numba paralelo CPU, Numba CUDA
si esta disponible y PyTorch para los dispositivos disponibles. Por defecto los
scripts se ejecutan uno por vez para evitar que compitan por recursos:

```bash
python benchmark_sobel.py --output=resultados_sobel.csv
```

Ejecutar varios scripts en paralelo acelera la recoleccion, pero contamina las
mediciones. Usarlo solo para una corrida exploratoria rapida:

```bash
python benchmark_sobel.py --workers=4
```

Para ejecutar PyTorch CPU y CUDA al regenerar el benchmark completo:

```bash
.venv-cuda/bin/python benchmark_sobel.py --devices=cpu,cuda --output=resultados_sobel.csv
```

El CSV incluye:

```text
method,device,image,runs,transferencia_promedio_s,rgb_gris_promedio_s,sobel_promedio_s,total_promedio_s,blancos_pct,speed_up,performance_pct
```

`transferencia_promedio_s` es `0` para CPU. Para CUDA y MPS mide el movimiento
de entrada y salida entre CPU y acelerador. `total_promedio_s` incluye ese costo:
`transferencia_promedio_s + rgb_gris_promedio_s + sobel_promedio_s`.

`speed_up` se calcula como `tiempo_total_secuencial / tiempo_total_metodo`.
`performance_pct` se calcula como `speed_up * 100`.
