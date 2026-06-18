import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import cv2
import os

img_size = (150, 150)
batch_size = 32
model_path = "cat_dog_model.keras"

# Data Augmentation
train_datagen = ImageDataGenerator(
    rescale=1./255,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True
)

test_datagen = ImageDataGenerator(rescale=1./255)

training_set = train_datagen.flow_from_directory(
    './data/training_set',
    target_size=img_size,
    batch_size=batch_size,
    class_mode='binary'
)

test_set = test_datagen.flow_from_directory(
    './data/test_set',
    target_size=img_size,
    batch_size=batch_size,
    class_mode='binary'
)

# Model
def create_model():
    model = keras.Sequential([
        keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(150, 150, 3)),
        keras.layers.MaxPooling2D((2, 2)),

        keras.layers.Conv2D(64, (3, 3), activation='relu'),
        keras.layers.MaxPooling2D((2, 2)),

        keras.layers.Conv2D(128, (3, 3), activation='relu'),
        keras.layers.MaxPooling2D((2, 2)),

        keras.layers.Flatten(),
        keras.layers.Dense(512, activation='relu'),
        keras.layers.Dense(1, activation='sigmoid')
    ])

    model.compile(
        optimizer='adam',
        loss='binary_crossentropy',
        metrics=['accuracy']
    )
    return model

# Load or Train
if os.path.exists(model_path):
    print("Loading existing model...")
    model = keras.models.load_model(model_path)
else:
    print("No model found. Training new model...")
    model = create_model()

    model.fit(
        training_set,
        epochs=20,
        validation_data=test_set
    )

    model.save(model_path)
    print("Model saved!")


# Prediction
print(training_set.class_indices)

classes = {v: k for k, v in training_set.class_indices.items()}

image = cv2.imread("./data/test_set/dogs/dog.4004.jpg")
image = cv2.resize(image, img_size)
image = np.expand_dims(image, axis=0)
image = image / 255.0

# make prediction 
predictions = model.predict(image)

class_index = np.argmax(predictions[0])
# define classes 
classes = ['cat','dog']

print(f"Model predicts that image is a : {classes[class_index]}")