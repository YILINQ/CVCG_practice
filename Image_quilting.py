import cv2
import numpy as np


def random_patch(texture, block_size):
    # suppose block is square
    h, w, c = texture.shape
    i, j = np.random.randint(h - block_size), np.random.randint(w - block_size)
    return texture[i: i + block_size, j:j + block_size]


def quilting(image, block_size, num_block, mode="random", sequence=False, smooth_factor=5):
    texture = cv2.imread(image)

    overlap = block_size // smooth_factor

    # calculate output image size
    h, w = block_size * num_block - (num_block - 1) * overlap, block_size * num_block - (num_block - 1) * overlap
    result = np.zeros((h, w, texture.shape[2]))

    for i in range(num_block):
        for j in range(num_block):
            x = j * (block_size - overlap)
            y = i * (block_size - overlap)
            # first patch is always random
            if i == 0 and j == 0:
                patch = random_patch(texture, block_size)
            else:
                if mode == "random":
                    patch = random_patch(texture, block_size)

            # dummy statement
            patch = random_patch(texture, block_size)
            result[y: y + block_size, x:x + block_size] = patch
    cv2.imshow("test", result)
    cv2.waitKey(1)


if __name__ == "__main__":
    quilting("strawberry.png", block_size=25, num_block=5)
