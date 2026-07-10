import numpy as np


def convolution2d(image, kernel, padding=0, stride=1):
    if image.ndim != 2:
        raise ValueError("Image must be 2D")

    if kernel.ndim != 2:
        raise ValueError("Kernel must be 2D")

    if stride <= 0:
        raise ValueError("Stride must be greater than zero")

    if padding < 0:
        raise ValueError("Padding must be positive")

    if padding > 0:
        image = np.pad(
            image,
            pad_width=padding,
            mode="constant",
            constant_values=0,
        )

    kernel_height, kernel_width = kernel.shape
    output_height = ((image.shape[0] - kernel_height) // stride) + 1
    output_width = ((image.shape[1] - kernel_width) // stride) + 1

    feature_map = np.zeros(
        (output_height, output_width),
        dtype=image.dtype
    )

    for ri in range(output_height):
        for ci in range(output_width):
            image_row = ri * stride
            image_col = ci * stride

            window = image[
                image_row:image_row + kernel_height,
                image_col:image_col + kernel_width
            ]

            result = window * kernel
            output = np.sum(result)

            feature_map[ri, ci] = output

    return feature_map


image = np.array([
    [1, 2, 3, 4, 5, 6, 7],
    [4, 5, 6, 7, 8, 9, 10],
    [7, 8, 9, 10, 11, 12, 13],
    [7, 8, 9, 10, 11, 12, 13],
    [7, 8, 9, 10, 11, 12, 13],
    [7, 8, 9, 10, 11, 12, 13],
    [7, 8, 9, 10, 11, 12, 13],
])

kernel = np.array([
    [1, 0, 1],
    [0, 1, 0],
    [1, 0, 1]
])

conv = convolution2d(image, kernel, stride=2)
print(conv)
