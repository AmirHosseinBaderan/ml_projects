import cv2
import numpy as np

def predict_image(model,image_path,img_size,class_indices):
    # invert mapping 
    classes = {v: k for k, v in class_indices.items()}
    
    # read image
    
    image = cv2.imread(image_path)
    image = cv2.resize(image, img_size)
    image = np.expand_dims(image, axis=0)
    image = image / 255.0
    
    # prediction
    prediction = model.predict(image)[0][0]
    print('raw prediction : ',prediction)
    
    # convert to class
    class_index = int(prediction > 0.5)
    result = classes[class_index]
    
    return result,prediction