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
        (3,3),
        activation='relu',
        input_shape=(150,150,3)
    ),
    
    tf.keras.layers.MaxPooling2D(),
    tf.keras.layers.Conv2D(
        64,
        (3,3),
        activation='relu'
    ),
    
    tf.keras.layers.MaxPooling2D(),
    tf.keras.layers.Conv2D(
        128,
        (3,3),
        activation='relu'
    ),
    
    tf.keras.layers.MaxPooling2D(),
    tf.keras.layers.Flatten(),
    
    tf.keras.layers.Dense(
        512,
        activation='relu'
    ),
    
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

history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=5
)

print(history.history.keys())

# show accuracy 
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])

plt.title('Model Accuracy')
plt.ylabel("Accuracy")
plt.xlabel('Epoch')
plt.legend(['Train','Validation'])

plt.show()

# show loss
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])

plt.title('Model Loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Validation'])

plt.show()