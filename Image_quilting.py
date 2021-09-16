import cv2
import numpy as np
import heapq
import PIL.Image as Image


def random_patch(texture, block_size):
    # suppose block is square
    h, w, c = texture.shape
    i, j = np.random.randint(h - block_size), np.random.randint(w - block_size)
    return texture[i: i + block_size, j:j + block_size]


def L2_diff(patch, block_size, overlap, res, y, x):
    error = 0
    if x > 0:
        error += np.sum((patch[:, :overlap] - res[y: y + block_size, x:x + overlap]) ** 2)
    if y > 0:
        error += np.sum((patch[:overlap, :] - res[y: y + overlap, x:x + block_size]) ** 2)
    if x > 0 and y > 0:
        error -= np.sum((patch[:overlap, :overlap] - res[y: y + overlap, x:x + overlap]) ** 2)
    return error


def best_patch(texture, block_size, overlap, y, x, res):
    h, w, c = texture.shape
    errors = np.zeros((h - block_size, w - block_size))

    for i in range(h - block_size):
        for j in range(w - block_size):
            patch = texture[i: i + block_size, j: j + block_size]
            error = L2_diff(patch, block_size, overlap, res, y, x)
            errors[i, j] = error
    i, j = np.unravel_index(np.argmin(errors), errors.shape)
    return texture[i: i + block_size, j:j + block_size]


def cut_path(errors):
    pq = [(error, [i]) for i, error in enumerate(errors[0])]
    heapq.heapify(pq)

    h, w = errors.shape
    visited = set()
    while pq:
        error, path = heapq.heappop(pq)
        current_length = len(path)
        current_index = path[-1]

        if current_length == h:
            return path
        for delta in range(-1, 2):
            next_index = current_index + delta
            if 0 <= next_index < w:
                if (current_length, next_index) not in visited:
                    cum_error = error + errors[current_length, next_index]
                    heapq.heappush(pq, (cum_error, path + [next_index]))
                    visited.add((current_length, next_index))
    return list(visited)

def min_cut_path(patch, overlap, result, y, x):
    patch = patch.copy()
    dy, dx, dc = patch.shape
    min_cut = np.zeros_like(patch, dtype=bool)

    if x > 0:
        left_error = np.sum((patch[:, :overlap] - result[y: y + dy, x:x + overlap]) ** 2, axis=2)
        for i, j in enumerate(cut_path(left_error)):
            min_cut[i, :j] = True
    if y > 0:
        top_error = np.sum((patch[:overlap, :] - result[y: y + overlap, x:x + dx]) ** 2, axis=2)
        for j, i in enumerate(cut_path(top_error.T)):
            min_cut[:i, j] = True

    np.copyto(patch, result[y:y + dy, x:x + dx], where=min_cut)
    return patch


def quilting(image, block_size, num_block, mode="random", smooth_factor=5):
    texture = cv2.imread(image, 3)
    overlap = block_size // smooth_factor

    # calculate output image size
    h, w = block_size * num_block - (num_block - 1) * overlap, block_size * num_block - (num_block - 1) * overlap
    result = np.zeros((h, w, texture.shape[-1]), np.float32)

    for i in range(num_block):
        for j in range(num_block):
            x = j * (block_size - overlap)
            y = i * (block_size - overlap)
            # first patch is always random
            if i == 0 and j == 0:
                patch = random_patch(texture, block_size)
            else:
                patch = random_patch(texture, block_size)
                if mode == "random":
                    patch = random_patch(texture, block_size)
                if mode == "best":
                    patch = best_patch(texture, block_size, overlap, y, x, result)
                if mode == "cut":
                    result = result.astype(np.uint8)
                    patch = best_patch(texture, block_size, overlap, y, x, result)
                    patch = min_cut_path(patch, overlap, result, y, x)
            # dummy statement
            result[y: y + block_size, x:x + block_size] = patch

            # cv2.imshow("result", result)
            # cv2.waitKey(0)
    result = result.astype(np.uint8)
    cv2.imshow("test", result)
    cv2.waitKey(0)


def main():
    # img_path, block_size, num_block, mode
    try:
        print(len(sys.argv))
        assert len(sys.argv) == 5
        quilting(image=sys.argv[1], block_size=int(sys.argv[2]), num_block=int(sys.argv[3]), mode=sys.argv[4])
    except:
        print("Usage: python image_quilting_texture.py [img_path] [block_size] [num_block] [mode]")


if __name__ == "__main__":
    import sys
    import os
    import argparse

    main()
