import tensorflow as tf
from tensorflow.keras.datasets import mnist 
from tensorflow.keras.utils import to_categorical
import numpy as np
import matplotlib.pyplot as plt

# load dataset
(x_train,y_train),(x_test,y_test) = mnist.load_data()

# dataset normalize 
# pixles between 0-255
x_train = x_train / 255
x_test = x_test / 255


y_train = to_categorical(y_train,10)
y_test = to_categorical(y_test,10)

# make model 
model = tf.keras.Sequential([
    tf.keras.layers.Flatten(input_shape=(28,28)),
    tf.keras.layers.Dense(128,activation='relu'),
    tf.keras.layers.Dense(64,activation='relu'),
    tf.keras.layers.Dense(10,activation='softmax')
])

model.compile(
    optimizer="adam", # -> Learning algorithm,
    loss="categorical_crossentropy" ,
    metrics=["accuracy"] 
)

# train 
model.fit(
    x_train,
    y_train,
    epochs=10,
    batch_size=100,
    validation_split=0.1
)

loss,accuracy = model.evaluate(x_test,y_test)

print(f"Accuracy : {accuracy}")

# predict 
predictions = model.predict(x_test,verbose=0)

wrong_images = []

print(f"total test data count : {len(y_test)}")
for i in range(len(y_test)):
    pred = np.argmax(predictions[i])
    true = np.argmax(y_test[i])
    
    if pred != true:
        wrong_images.append((i,pred,true))
        
current = 0
fig,ax = plt.subplots()

def show_image():
    ax.clear()
    
    idx,pred,true = wrong_images[current]
    ax.imshow(x_test[idx],cmap='gray_r')
    
    ax.set_title(
        f"{current + 1}/{len(wrong_images)} | Prediction : {pred} Label : {true}"
    )
    ax.axis('off')
    
    fig.canvas.draw()
    
def on_key(event):
    global current
    
    if event.key == 'right':
        current = min(current + 1,len(wrong_images) - 1)
    elif event.key == "left":
        current = max(current -1 ,0)
    
    show_image()
    
fig.canvas.mpl_connect("key_press_event",on_key)

show_image()
plt.show()