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

vertical_kernel = np.array([
    [-1, 0, 1],
    [-1, 0, 1],
    [-1, 0, 1]
])

horizontal_kernel = np.array([
    [ 1, 1, 1],
    [ 0, 0, 0],
    [-1,-1,-1]
])

blur_kernel = np.array([
    [1,1,1],
    [1,1,1],
    [1,1,1]
]) / 9

sharpen_kernel = np.array([
    [ 0,-1, 0],
    [-1, 5,-1],
    [ 0,-1, 0]
])

fm = convolution2d(image, kernel)
vfm = convolution2d(image, vertical_kernel)
hfm = convolution2d(image, horizontal_kernel)
bfm = convolution2d(image, blur_kernel)
sfm = convolution2d(image, sharpen_kernel)

print("-" * 20)
print("Kernel")
print(fm)

print("-" * 20)
print("Vertical kernel")
print(vfm)

print("-" * 20)
print("Horizontal kernel")
print(hfm)

print("-" * 20)
print("Blur kernel")
print(bfm)

print("-" * 20)
print("Sharpen kernel")
print(sfm)

