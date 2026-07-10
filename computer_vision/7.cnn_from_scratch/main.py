from convolution import convolution2d
from pooling import max_pool2d
from activation import relu

import numpy as np

"""
              6×6 Image
                   │
                   ▼
        +--------------------+
        |   Convolution      |
        +--------------------+
                   │
                   ▼
              Feature Map
                   │
                   ▼
        +--------------------+
        |      ReLU          |
        +--------------------+
                   │
                   ▼
        +--------------------+
        |   Max Pooling      |
        +--------------------+
                   │
                   ▼
          Smaller Feature Map
                   │
                   ▼
        +--------------------+
        |     Flatten        |
        +--------------------+
                   │
                   ▼
             1D Vector
                   │
                   ▼
              Dense Layer
"""

# pipeline
"""
                 Image
                   │
                   ▼
          convolution2d()
                   │
                   ▼
                ReLU
                   │
                   ▼
            max_pool2d()
                   │
                   ▼
               Flatten
                   │
                   ▼
            Dense Layer
                   │
                   ▼
                ReLU
                   │
                   ▼
               Softmax
                   │
                   ▼
             Final Prediction
"""

image = np.array([
    [1, 2, 3, 4, 5, 6],
    [7, 8, 9, 10, 11, 12],
    [13, 14, 15, 16, 17, 18],
    [19, 20, 21, 22, 23, 24],
    [25, 26, 27, 28, 29, 30],
    [31, 32, 33, 34, 35, 36]
])

print("-" * 40)
print("Original Image")
print(image)

kernel = np.array([
    [1, 0, 1],
    [0, 1, 0],
    [1, 0, 1]
])

conv = convolution2d(
    image,
    kernel,
    padding=1,
    stride=1
)

print("-" * 40)
print("After Convolution")
print(conv)

relu_output = relu(conv)
print(relu_output)

pool = max_pool2d(
    relu_output,
    pool_size=2,
    stride=2
)

print("-" * 40)
print("After MaxPooling")
print(pool)

flatten = pool.flatten()

print("-" * 40)
print("Flatten")
print(flatten)

input_size = flatten.shape[0]
output_size = 3

weights = np.random.randn(input_size, output_size)

bias = np.random.randn(output_size)

dense_output = np.dot(flatten, weights) + bias
print("-" * 40)
print("Dense output")
print(dense_output)

dense_relu = relu(dense_output)
print("-" * 40)
print("Dense Relu")
print(dense_relu)

exp = np.exp(dense_relu)
softmax = exp / np.sum(exp)
print("-" * 40)
print("Softmax")
print(softmax)
