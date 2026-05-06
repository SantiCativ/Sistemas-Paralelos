from PIL import Image
import numpy as np
import matplotlib.pyplot as plt

# cargar imagen
image = Image.open("imagenes/penguins.jpg")

# RGB -> numpy
rgb = np.asarray(image, dtype=np.float32)

# convertir a gris (vectorizado)
gray = 0.299 * rgb[:, :, 0] + 0.587 * rgb[:, :, 1] + 0.114 * rgb[:, :, 2]
gray = np.clip(gray, 0, 255).astype(np.float32)

# Sobel con slices vectorizados
# Cada vecino de la vecindad 3x3 se extrae como un subarreglo desplazado.
# Esto evita cualquier loop explícito en Python: todas las operaciones
# se aplican sobre arreglos completos usando broadcasting de NumPy.

top_left     = gray[:-2, :-2]
top          = gray[:-2, 1:-1]
top_right    = gray[:-2, 2:]
left         = gray[1:-1, :-2]
right        = gray[1:-1, 2:]
bottom_left  = gray[2:,  :-2]
bottom       = gray[2:,  1:-1]
bottom_right = gray[2:,  2:]

# Gx: kernel [[-1,0,1],[-2,0,2],[-1,0,1]]
gx = (-top_left  + top_right
      - 2.0*left + 2.0*right
      - bottom_left + bottom_right)

# Gy: kernel [[-1,-2,-1],[0,0,0],[1,2,1]]
gy = (-top_left  - 2.0*top - top_right
      + bottom_left + 2.0*bottom + bottom_right)

# magnitud del gradiente
result = np.zeros_like(gray)
result[1:-1, 1:-1] = np.sqrt(gx**2 + gy**2)

# mostrar resultado
plt.imshow(result, cmap="gray")
plt.axis("off")
plt.show()
