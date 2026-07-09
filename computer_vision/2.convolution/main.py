import numpy as np

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

window = image[:3, :3]
print("Window")
print(window)
print("-" * 20)

result = window * kernel
print(result)
print("Result")
print("-" * 20)

output = np.sum(result)
print("Output")
print(output)

kernel_size = 3

for ri in range(image.shape[0] - kernel_size + 1):
    for ci in range(image.shape[1] - kernel_size + 1):
        window = image[ri:ri+kernel_size,
                       ci:ci+kernel_size]

        print(window)
        print("-" * 20)
