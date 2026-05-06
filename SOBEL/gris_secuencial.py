from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
import time

# cargar imagen
image = Image.open('messi.jpeg')

# convertir a matriz
image = np.array(image)

alto, ancho, canales = image.shape


inicio_tiempo = time.time()

# recorrer píxel por píxel
for i in range(alto):
    for j in range(ancho):

        # convertir a int para evitar overflow
        r = int(image[i][j][0])
        g = int(image[i][j][1])
        b = int(image[i][j][2])

        # calcular gris
        gris = (r + g + b) // 3

        # reemplazar pixel
        image[i][j] = [gris, gris, gris]


fin_tiempo = time.time()

print(f"Tiempo secuencial: {fin_tiempo - inicio_tiempo:.4f} segundos")

# mostrar imagen
plt.imshow(image)

print(image.shape)

plt.show()