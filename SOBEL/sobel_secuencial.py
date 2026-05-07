from PIL import Image

from sobel_common import run_cli


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


def build_runner():
    def load_step(image_path):
        # cargar imagen
        image = Image.open(image_path)

        # dimensiones
        width, height = image.size

        # convertir imagen a matriz RGB 2D
        pixels = list(image.getdata())

        return pixels, width, height

    def gray_step(source):
        pixels, width, height = source

        # convertir imagen a matriz RGB 2D
        rgb = [pixels[i * width : (i + 1) * width] for i in range(height)]

        # convertir a gris
        return rgb_to_gray(rgb), width, height

    def sobel_step(gray_data):
        gray, width, height = gray_data

        # aplicar sobel
        return sobel_sequential(gray, width, height)

    return {"load": load_step, "gray": gray_step, "sobel": sobel_step}


if __name__ == "__main__":
    run_cli(build_runner(), "secuencial")
