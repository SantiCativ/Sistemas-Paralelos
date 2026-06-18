import cv2
import numpy as np
import time

KERNEL = np.array([[-2, -1, 0], [-1, 1, 1], [0, 1, 2]], dtype=np.float32)


def emboss_naive(image):
    h, w, c = image.shape

    result = np.zeros_like(image, dtype=np.float32)

    for channel in range(c):

        for y in range(1, h - 1):

            for x in range(1, w - 1):

                region = image[y - 1 : y + 2, x - 1 : x + 2, channel]

                result[y, x, channel] = np.sum(region * KERNEL) + 128

    return np.clip(result, 0, 255).astype(np.uint8)


cap = cv2.VideoCapture("paisaje.mp4")

ok, frame = cap.read()

cap.release()

if not ok:
    raise RuntimeError("No se pudo leer el frame")

inicio = time.perf_counter()

emboss_naive(frame)

fin = time.perf_counter()

print(f"Tiempo frame: {fin - inicio:.3f} s")
