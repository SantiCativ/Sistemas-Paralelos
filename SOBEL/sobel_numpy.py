from PIL import Image
import numpy as np
import matplotlib.pyplot as plt

# cargar imagen
image = Image.open("imagenes/penguins.jpg")

# RGB -> numpy
rgb = np.asarray(image, dtype=np.float32)

# convertir a gris
gray = 0.299 * rgb[:, :, 0] + 0.587 * rgb[:, :, 1] + 0.114 * rgb[:, :, 2]

gray_u8 = np.clip(gray, 0, 255).astype(np.uint8)

# Sobel
img = gray_u8.astype(np.float32)

# dimensiones
height, width = img.shape

# kernels
Gx = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=np.float32)

Gy = np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]], dtype=np.float32)

# imagen resultado
result = np.zeros((height, width), dtype=np.float32)

# recorrer imagen
for row in range(1, height - 1):

    for col in range(1, width - 1):

        # ventana 3x3
        p = img[row - 1 : row + 2, col - 1 : col + 2]

        # convolución
        gx = np.sum(p * Gx)
        gy = np.sum(p * Gy)

        # magnitud gradiente
        result[row, col] = np.sqrt(gx * gx + gy * gy)

# mostrar resultado
plt.imshow(result, cmap="gray")
plt.axis("off")
plt.show()
