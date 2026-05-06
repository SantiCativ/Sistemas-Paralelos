from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
import threading
import time
import sys


# ==========================================
# FUNCION THREAD
# ==========================================

def convertir_a_gris(image, inicio, fin, ancho):

    for i in range(inicio, fin):

        for j in range(ancho):

            r = int(image[i, j, 0])
            g = int(image[i, j, 1])
            b = int(image[i, j, 2])

            gris = (r + g + b) // 3

            image[i, j, 0] = gris
            image[i, j, 1] = gris
            image[i, j, 2] = gris


# ==========================================
# CARGAR IMAGEN
# ==========================================

image = Image.open("messi.jpeg")

image = np.array(image)

alto, ancho, canales = image.shape

# ==========================================
# THREADING
# ==========================================

cantidad_hilos = 16

filas_por_hilo = alto // cantidad_hilos

threads = []

inicio_tiempo = time.time()

for h in range(cantidad_hilos):

    inicio = h * filas_por_hilo

    if h == cantidad_hilos - 1:
        fin = alto
    else:
        fin = inicio + filas_por_hilo

    t = threading.Thread(
        target=convertir_a_gris,
        args=(image, inicio, fin, ancho)
    )

    threads.append(t)

    t.start()

# esperar threads
for t in threads:
    t.join()

fin_tiempo = time.time()

print(f"Tiempo: {fin_tiempo - inicio_tiempo:.4f} segundos")

# ==========================================
# MOSTRAR
# ==========================================

plt.imshow(image)

plt.title("Escala de grises - threading no GIL")

plt.show()