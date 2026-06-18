import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import os
from predictor import Predictor
import sys

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


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python3 main.py <image_path | folder_path>")
        sys.exit(1)
        
    predictor = Predictor(model,training_set.class_indices,img_size)
    input_path = sys.argv[1]
    if os.path.isdir(input_path):
        print("Folder detected")
        cat_count,dog_count = predictor.predict_folder(input_path)
        print(f"Cat count : {cat_count} / Dog count {dog_count}")
        
    elif os.path.isfile(input_path):
        print("Image detected")
        result,_ = predictor.predict_image(input_path)
        
        print(f"Prediction : {result}")
    else:
        print("Invalid path")