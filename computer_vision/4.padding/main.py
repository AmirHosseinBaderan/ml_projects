import numpy as np


def convolution2d(image, kernel, padding=0):
    if padding > 0:
        image = np.pad(
            image,
            pad_width=padding,
            mode="constant",
            constant_values=0,
        )

    kernel_height, kernel_width = kernel.shape
    output_height = image.shape[0] - kernel_height + 1
    output_width = image.shape[1] - kernel_width + 1
    feature_map = np.zeros(
        (output_height, output_width),
        dtype=image.dtype
    )

    for ri in range(output_height):
        for ci in range(output_width):
            window = image[
                ri:ri + kernel_height,
                ci:ci + kernel_width
            ]

            result = window * kernel
            output = np.sum(result)

            feature_map[ri, ci] = output

    return feature_map


image = np.array([
    [1, 2, 3],
    [4, 5, 6],
    [7, 8, 9]
])

padded = np.pad(
    image,
    pad_width=1,
    mode="constant",
    constant_values=0,
)

print(padded)
