from pathlib import Path
import matplotlib.pyplot as plt
import tensorflow as tf
from model_plt import show_model_plot
import os
from predictor import predict_image

MODEL_PATH = "cats_vs_dogs.keras"

base_dir = Path('./data')
cat_dir = base_dir / 'train/cats/'
dog_dir = base_dir / 'train/dogs/'
cat_test_dir = base_dir / 'validation/cats/'
dog_test_dir = base_dir / 'validation/dogs/'

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
    image_size=(IMG_HEIGHT, IMG_WIDTH),
    batch_size=BATCH_SIZE
)

val_ds = tf.keras.utils.image_dataset_from_directory(
    base_dir / 'validation',
    image_size=(IMG_HEIGHT, IMG_WIDTH),
    batch_size=BATCH_SIZE
)

AUTOTUNE = tf.data.AUTOTUNE

train_ds = train_ds.cache().shuffle(1000).prefetch(buffer_size=AUTOTUNE)
val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)

# normalize
normalization_layer = tf.keras.layers.Rescaling(1.0 / 255)

# model builder function (important for reuse)
def build_model():
    return tf.keras.Sequential([
        normalization_layer,

        tf.keras.layers.Conv2D(32,3,activation='relu'),
        tf.keras.layers.MaxPooling2D(),

        tf.keras.layers.Conv2D(64,3,activation='relu'),
        tf.keras.layers.MaxPooling2D(),

        tf.keras.layers.Conv2D(128,3,activation='relu'),
        tf.keras.layers.MaxPooling2D(),

        tf.keras.layers.Flatten(),

        tf.keras.layers.Dense(512, activation='relu'),

        tf.keras.layers.Dense(1, activation='sigmoid')
    ])

# LOAD OR TRAIN
if os.path.exists(MODEL_PATH):
    print("Loading saved model...")
    model = tf.keras.models.load_model(MODEL_PATH)
    history = None

else:
    print("Training new model...")

    model = build_model()

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

    model.save(MODEL_PATH)
    print("Model saved!")

# EVALUATION
loss, acc = model.evaluate(val_ds)
print(f'loss : {loss} , acc : {acc}')

print("model summary:")
model.summary()

# PLOTS (only if trained)
if history is not None:
    print(history.history.keys())
    print(max(history.history['accuracy']))
    print(max(history.history['val_accuracy']))

    show_model_plot(history)
    
for images, labels in val_ds.take(1):
    preds = model.predict(images)

    print("raw predictions:")
    print(preds[:10])

    print("labels:")
    print(labels.numpy()[:10])

    
print("Check dogs image")
wrong_dogs = 0
for d in range(len(list(dog_test_dir.glob('*')))):
    predict,lable = predict_image(list(dog_test_dir.glob('*'))[d],model,['cats','dogs'])
    if lable != 'dogs':
        wrong_dogs +=1 

print(f"Wrong dogs detection : {wrong_dogs}")

print("Check cats image")
wrong_cats = 0
for d in range(len(list(cat_test_dir.glob('*')))):
    predict,lable = predict_image(list(cat_test_dir.glob('*'))[d],model,['cats','dogs'])
    if lable != 'cats':
        wrong_cats +=1 

print(f"Wrong cats detection : {wrong_cats}")