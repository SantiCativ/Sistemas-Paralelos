# Resumen TP4 - Video Emboss

| Implementacion | Frames | Resolucion | FPS original | FPS efectivos | Lectura (s) | Filtrado (s) | Escritura (s) | CPU->GPU (s) | GPU->CPU (s) | Total (s) | RAM MB | GPU MB | Codec | Speedup |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|---:|
| Secuencial | 1177 | 3840x2160 | 30.000 | 3.153 | 3.219415 | 318.167310 | 51.770654 | 0.000000 | 0.000000 | 373.342208 | 1146.72 | 0.00 | mp4v + ffmpeg audio | 1.0000 |
| PyTorch CPU | 1177 | 3840x2160 | 30.000 | 4.264 | 3.310254 | 188.184774 | 84.362383 | 0.000000 | 0.000000 | 276.049843 | 1192.01 | 0.00 | mp4v + ffmpeg audio | 1.3524 |
| PyTorch CUDA | 1177 | 3840x2160 | 30.000 | 11.410 | 3.305360 | 7.629487 | 82.227037 | 5.040686 | 4.537871 | 103.156848 | 1396.65 | 309.84 | mp4v + ffmpeg audio | 3.6192 |
