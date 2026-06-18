import tensorflow as tf
import numpy as np

IMG_SIZE = (150,150)

def predict_image(image_path,model):
    img = tf.keras.utils.load_img(
        image_path,
        target_size=IMG_SIZE
    )
    
    img_array = tf.keras.utils.img_to_array(img)
    img_array = img_array / 255 # normalize
    img_array = np.expand_dims(img_array,axis=0) # (1,150,150,3)
    
    prediction = model.predict(img_array,verbose=0)[0][0]
    if prediction > 0.5:
        label = "Dog"
    else:
        label = "Cat"
    
    print(f"Prediction Score : {prediction:.4f}")
    print(f"Result {label}")