"""
Filters (Kernel)
If convolution is "how to look," the filter determines "what to look for."
"""

import numpy as np


def convolution2d(image, kernel):
    kernel_size = kernel.shape[0]
    output_height = image.shape[0] - kernel_size + 1
    output_width = image.shape[1] - kernel_size + 1
    feature_map = np.zeros((output_height, output_width))

    for ri in range(output_height):
        for ci in range(output_width):
            window = image[
                ri:ri + kernel_size,
                ci:ci + kernel_size
            ]

            result = window * kernel
            output = np.sum(result)

            feature_map[ri, ci] = output

    return feature_map


image = np.array([
    [1, 2, 3, 4, 5],
    [6, 7, 8, 9, 10],
    [11, 12, 13, 14, 15],
    [16, 17, 18, 19, 20],
    [21, 22, 23, 24, 25],
    [21, 22, 23, 24, 25],
])

print("Image")
print(image)
print("-" * 20)

kernel = np.array([
    [1, 0, 1],
    [0, 1, 0],
    [1, 0, 1]
])
print("Kernel")
print(kernel)
print("-" * 20)

fm = convolution2d(image, kernel)
print(fm)
