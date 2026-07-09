import numpy as np

gray_image = np.random.randint(
    0,
    256,
    (28, 28),
    dtype=np.uint8
)

print(gray_image.shape)

color_image = np.random.randint(
    0,
    256,
    (224, 224, 3),
    dtype=np.uint8
)
print(color_image.shape)
