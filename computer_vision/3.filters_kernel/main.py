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
    [0,0,0,255,255,255],
    [0,0,0,255,255,255],
    [0,0,0,255,255,255],
    [0,0,0,255,255,255],
    [0,0,0,255,255,255],
    [0,0,0,255,255,255]
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


kernels = {
    "Vertical": vertical_kernel,
    "Horizontal": horizontal_kernel,
    "Blur": blur_kernel,
    "Sharpen": sharpen_kernel
}

for name, kernel in kernels.items():

    print("=" * 40)
    print(name)
    print("=" * 40)

    feature_map = convolution2d(image, kernel)

    print(feature_map)