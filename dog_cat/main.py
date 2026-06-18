from pathlib import Path
import matplotlib.pyplot as plt
import tensorflow as tf
from model_plt import show_model_plot

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

AUTOTUNE = tf.data.AUTOTUNE
train_ds = train_ds.cache().shuffle(1000).prefetch(
    buffer_size=AUTOTUNE
)

val_ds = val_ds.cache().prefetch(
    buffer_size=AUTOTUNE
)

# normalize ds
normalization_layer = tf.keras.layers.Rescaling(1.0/ 255)

# Create model 
model = tf.keras.Sequential([
    normalization_layer,

    tf.keras.layers.Conv2D(
        32,
        3,
        activation='relu'
    ),
    tf.keras.layers.MaxPooling2D(),

    tf.keras.layers.Conv2D(
        64,
        3,
        activation='relu'
    ),
    tf.keras.layers.MaxPooling2D(),

    tf.keras.layers.Conv2D(
        128,
        3,
        activation='relu'
    ),
    tf.keras.layers.MaxPooling2D(),

    tf.keras.layers.GlobalAveragePooling2D(),

    tf.keras.layers.Dense(
        256,
        activation='relu'
    ),

    tf.keras.layers.Dropout(0.2), 

    tf.keras.layers.Dense(
        1,
        activation='sigmoid'
    )
])

model.compile(
    optimizer='adam',
    loss='binary_crossentropy',
    metrics=['accuracy']
)

early_stop = tf.keras.callbacks.EarlyStopping(
    monitor='val_loss',
    patience=3,
    restore_best_weights=True
)

history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=20,
    callbacks=[early_stop]
)

# remove after check model
loss, acc = model.evaluate(val_ds)

print(f'loss : {loss} , acc : {acc}')

model.save("cats_vs_dogs.keras")

print(history.history.keys())
print(max(history.history['accuracy']))
print(max(history.history['val_accuracy']))

model.summary()

# show accuracy 
show_model_plot(history)