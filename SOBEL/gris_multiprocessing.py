from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
import multiprocessing as mp
import time


# ==========================================
# FUNCION DEL WORKER
# ==========================================

def convertir_bloque_a_gris(bloque):

    alto, ancho, canales = bloque.shape

    for i in range(alto):

        for j in range(ancho):

            r = int(bloque[i, j, 0])
            g = int(bloque[i, j, 1])
            b = int(bloque[i, j, 2])

            gris = (r + g + b) // 3

            bloque[i, j, 0] = gris
            bloque[i, j, 1] = gris
            bloque[i, j, 2] = gris

    return bloque


# ==========================================
# MAIN
# ==========================================

if __name__ == "__main__":

    # cargar imagen
    image = Image.open("messi.jpeg")

    # convertir a numpy
    image = np.array(image)

    print("Shape:", image.shape)

    # cantidad de procesos
    cantidad_procesos = 4

    # ==========================================
    # DIVIDIR IMAGEN EN BLOQUES
    # ==========================================

    bloques = np.array_split(image, cantidad_procesos, axis=0)

    # ==========================================
    # PROCESAMIENTO PARALELO
    # ==========================================

    inicio = time.time()

    with mp.Pool(processes=cantidad_procesos) as pool:

        resultados = pool.map(convertir_bloque_a_gris, bloques)

    fin = time.time()

    print(f"Tiempo multiprocessing: {fin - inicio:.4f} segundos")

    # ==========================================
    # RECONSTRUIR IMAGEN
    # ==========================================

    image_gris = np.vstack(resultados)

    # ==========================================
    # MOSTRAR
    # ==========================================

    plt.imshow(image_gris)

    plt.title("Imagen en gris - multiprocessing")

    plt.show()