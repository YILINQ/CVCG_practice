import numpy as np
import cv2


def image_erosion(image, filter):
    assert type(image) == np.ndarray
    assert filter[0] == filter[1]
    assert image.shape[2] == 1
    h, w, _ = image.shape
    padding = filter[0] // 2
    for i in range(padding, h - padding):
        for j in range(padding, w - padding):
            if (np.array(image[j - padding: j + padding, i - padding: i + padding]) == 0).any():
                image[i][j] = 0


def image_dilation(image, filter):
    assert type(image) == np.ndarray
    assert filter[0] == filter[1]
    assert image.shape[2] == 1
    h, w, _ = image.shape
    padding = filter[0] // 2
    for i in range(padding, h - padding):
        for j in range(padding, w - padding):
            if (np.array(image[j - padding: j + padding, i - padding: i + padding]) == 1).all():
                image[i][j] = 1
