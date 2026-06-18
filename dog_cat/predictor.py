import tensorflow as tf
import numpy as np

IMG_SIZE = (150,150)

def predict_image(image_path, model, class_names):
    img = tf.keras.utils.load_img(image_path, target_size=(150,150))
    img_array = tf.keras.utils.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = img_array / 255.0

    prediction = model.predict(img_array, verbose=0)[0][0]

    label = class_names[int(prediction > 0.5)]

    return prediction, label