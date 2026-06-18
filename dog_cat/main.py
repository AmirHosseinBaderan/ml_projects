import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import cv2

img_size = (150,150)
batch_size = 32

# Createing ImageDataGenerator
train_datagen = ImageDataGenerator(rescale = 1./255,
                                   shear_range=0.2,
                                   zoom_range=0.2,
                                   horizontal_flip=True)

test_dategen = ImageDataGenerator(rescale=1./255)

# Createing training and testing sets 
training_set = train_datagen.flow_from_directory('./data/training_set',
                                                target_size=img_size,
                                                batch_size=batch_size,
                                                class_mode='binary')

test_set = test_dategen.flow_from_directory('./data/test_set',
                                            target_size=img_size,
                                            batch_size=batch_size,
                                            class_mode='binary')

# define model architecture 
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

# Compile model 
model.compile(loss='binary_crossentropy',optimizer='adam',metrics=['accuracy'])
model.fit(training_set,epochs=50,validation_data=0.1)

# read image to be tested 
image = cv2.imread("test_image.jpg")
# Resize the image to the input shape of the model
image = cv2.resize(image, (150, 150))
# convert image to numpy array and add an additional dimension
image = np.expand_dims(image,axis=0)

# Normalize image 
image = image / 255.0

# make prediction 
predictions = model.predict(image)

class_index = np.argmax(predictions[0])
# define classes 
classes = ['cat','dog']

print(f"Model predicts that image is a : {classes[class_index]}")