from pathlib import Path
import matplotlib.pyplot as plt
import tensorflow as tf

base_dir = Path('./data')
cat_dir = base_dir / 'train/cats/'
dog_dir = base_dir / 'train/dogs/'

train_cats = len(list(cat_dir.glob('*')))
train_dogs = len(list(dog_dir.glob('*')))

print(f"Cats : {train_cats}")
print(f"Dogs : {train_dogs}")

# Create dataset
IMG_HEIGHT = 150
IMG_WIDTH = 150
BATCH_SIZE = 32

train_ds = tf.keras.utils.image_dataset_from_directory(
    base_dir / 'train',
    image_size=(IMG_HEIGHT,IMG_WIDTH),
    batch_size=BATCH_SIZE
)

# validation data
val_ds = tf.keras.utils.image_dataset_from_directory(
    base_dir / 'validation',
    image_size=(IMG_HEIGHT,IMG_WIDTH),
    batch_size=BATCH_SIZE
)