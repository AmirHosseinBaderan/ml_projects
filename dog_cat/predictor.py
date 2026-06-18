import cv2
import numpy as np
import os

class Predictor:
    def __init__(self,model,class_indices,img_size):
        self.model = model
        self.class_indices = class_indices
        self.img_size = img_size
    
    def predict_image(self,image_path):
        # invert mapping 
        classes = {v: k for k, v in self.class_indices.items()}

        # read image

        image = cv2.imread(image_path)
        image = cv2.resize(image, self.img_size)
        image = np.expand_dims(image, axis=0)
        image = image / 255.0

        # prediction
        prediction = self.model.predict(image)[0][0]
        print('raw prediction : ',prediction)

        # convert to class
        class_index = int(prediction > 0.5)
        result = classes[class_index]

        return result,prediction

    def predict_folder(self,folder_path):
        cat_count = 0
        dog_count = 0
        
        for file in os.listdir(folder_path):
            path = os.path.join(folder_path,file)
            
            if not os.path.isfile(path):
                continue
            
            result,_ = self.predict_image(path)
            
            if result == "cats":
                cat_count += 1
            elif result == "dogs":
                dog_count += 1
        
        return cat_count,dog_count 