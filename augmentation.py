import cv2
import random
import numpy as np


def augment(img):
    r, c, d = img.shape
    num_transformation = 3
    num_times = 5
    res = np.empty((r, c, d, num_transformation * num_times))
    for i in range(num_times):
        res[:, :, :, num_transformation * i] = img
        res[:, :, :, num_transformation * i + 1] = brightness(img, 0.9, 2)
        res[:, :, :, num_transformation * i + 2] = channel_shift(img, 10)

    return res


# img = channel_shift(img, 60)
# img = brightness(img, 0.5, 3)


def brightness(img, low, high):
    value = random.uniform(low, high)
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    hsv = np.array(hsv, dtype=np.float64)
    hsv[:, :, 1] = hsv[:, :, 1]*value
    hsv[:, :, 1][hsv[:, :, 1] > 255] = 255
    hsv[:, :, 2] = hsv[:, :, 2]*value
    hsv[:, :, 2][hsv[:, :, 2] > 255] = 255
    hsv = np.array(hsv, dtype=np.uint8)
    img = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
    return img


def channel_shift(img, value):
    value = int(random.uniform(-value, value))
    img = img + value
    img[:, :, :][img[:, :, :] > 255] = 255
    img[:, :, :][img[:, :, :] < 0] = 0
    img = img.astype(np.uint8)
    return img
