# TP4 Video Emboss

Trabajo Practico de Sistemas Paralelos para procesar un video frame por frame con
un filtro Emboss implementado en tres variantes comparables:

- Secuencial CPU con NumPy.
- PyTorch CPU.
- PyTorch CUDA.

El pipeline usa `cv2.VideoCapture` y `cv2.VideoWriter`, por lo que no carga el
video completo en memoria. Si `ffmpeg` esta instalado, tambien puede extraer y
reincorporar audio al video final.

## Estructura

```text
tp4_video/
├── emboss_common.py
├── emboss_sequential.py
├── emboss_pytorch.py
├── video_processing_common.py
├── run_sequential.py
├── run_pytorch_cpu.py
├── run_pytorch_cuda.py
├── benchmark.py
├── requirements.txt
└── README.md
```

## Requisitos

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r tp4_video/requirements.txt
```

Para CUDA se necesita una instalacion de PyTorch compatible con la placa y los
drivers instalados. En Linux con RTX 4060 conviene instalar PyTorch siguiendo el
comando oficial para la version CUDA disponible en el equipo.

`ffmpeg` es opcional y solo se usa para preservar audio:

```bash
sudo apt install ffmpeg
```

## Ejecucion completa

Desde la raiz del repositorio:

```bash
python tp4_video/benchmark.py --input /ruta/al/video_4k.mp4
```

El benchmark genera:

- `outputs/emboss_sequential.mp4`
- `outputs/emboss_pytorch_cpu.mp4`
- `outputs/emboss_pytorch_cuda.mp4`, si CUDA esta disponible
- `outputs/resultados_tp4.csv`
- `outputs/resumen_tp4.md`

Para una prueba rapida con pocos frames:

```bash
python tp4_video/benchmark.py --input /ruta/al/video_4k.mp4 --limit-frames 30 --no-audio
```

## Scripts individuales

```bash
python tp4_video/run_sequential.py --input /ruta/al/video_4k.mp4
python tp4_video/run_pytorch_cpu.py --input /ruta/al/video_4k.mp4
python tp4_video/run_pytorch_cuda.py --input /ruta/al/video_4k.mp4
```

Opciones utiles:

```bash
--output outputs/salida.mp4
--codec mp4v
--limit-frames 120
--no-audio
```

## Metricas exportadas

El CSV contiene las columnas obligatorias de la consigna:

```text
implementacion,frames,width,height,fps_original,fps_efectivos,
tiempo_lectura,tiempo_filtrado,tiempo_escritura,tiempo_cpu_gpu,
tiempo_gpu_cpu,tiempo_total,ram_mb,gpu_mb,codec,speedup
```

`tiempo_total` incluye lectura, filtrado, escritura y, para CUDA, las
transferencias CPU->GPU y GPU->CPU. Los tiempos de audio por `ffmpeg` no se
mezclan con esas metricas para mantener comparable el pipeline de video.

## Filtro Emboss

El kernel usado en las tres variantes es:

```text
[-2, -1, 0]
[-1,  1, 1]
[ 0,  1, 2]
```

Luego de aplicar el kernel por canal se suma `128` y se limita el resultado al
rango `[0, 255]`. La version PyTorch usa `torch.nn.functional.conv2d` con
`groups=3`, de modo que cada canal se procesa independientemente.
