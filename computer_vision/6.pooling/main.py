import numpy as np

image = np.array([
    [1, 2, 3, 4],
    [5, 6, 7, 8],
    [9,10,11,12],
    [13,14,15,16]
])

def max_pool2d(image, pool_size=2, stride=2):
    output_height = ((image.shape[0] - pool_size) // stride) + 1
    output_width = ((image.shape[1] - pool_size) // stride) + 1

    feature_map = np.zeros(
        (output_height, output_width),
        dtype=image.dtype
    )

    for ri in range(output_height):
        for ci in range(output_width):
            image_row = ri * stride
            image_col = ci * stride

            window = image[
                image_row:image_row + pool_size,
                image_col:image_col + pool_size
            ]

            output = np.max(window)
            feature_map[ri, ci] = output

    return feature_map

pool = max_pool2d(image)
print(pool)