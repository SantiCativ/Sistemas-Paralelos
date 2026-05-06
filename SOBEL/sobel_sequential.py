from PIL import Image
import matplotlib.pyplot as plt


def rgb_to_gray(rgb):

    gray = []

    for row in rgb:

        gray_row = []

        for r, g, b in row:

            i = int(0.299 * r + 0.587 * g + 0.114 * b)

            gray_row.append(0 if i < 0 else 255 if i > 255 else i)

        gray.append(gray_row)

    return gray


def sobel_sequential(gray, width, height):

    kernel_x = [
        [-1, 0, 1],
        [-2, 0, 2],
        [-1, 0, 1],
    ]

    kernel_y = [
        [-1, -2, -1],
        [0, 0, 0],
        [1, 2, 1],
    ]

    result = [[0.0 for _ in range(width)] for _ in range(height)]

    for row in range(1, height - 1):

        for col in range(1, width - 1):

            gx = 0.0
            gy = 0.0

            for kr in range(3):

                for kc in range(3):

                    pixel = gray[row + kr - 1][col + kc - 1]

                    gx += pixel * kernel_x[kr][kc]
                    gy += pixel * kernel_y[kr][kc]

            result[row][col] = abs(gx) + abs(gy)

    return result


# cargar imagen
image = Image.open("imagenes/penguins_6000x6000.jpg")

# dimensiones
width, height = image.size

# convertir imagen a matriz RGB 2D
pixels = list(image.getdata())

rgb = [pixels[i * width : (i + 1) * width] for i in range(height)]

# convertir a gris
gray = rgb_to_gray(rgb)

# aplicar sobel
result = sobel_sequential(gray, width, height)

# mostrar
plt.imshow(result, cmap="gray")
plt.show()
