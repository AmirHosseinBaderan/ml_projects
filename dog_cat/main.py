import tensorflow as tf

url = "https://storage.googleapis.com/mledu-datasets/cats_and_dogs_filtered.zip"

path = tf.keras.utils.get_file(
    "cats_and_dogs.zip",
    origin=url,
    extract=True
)

print(path)