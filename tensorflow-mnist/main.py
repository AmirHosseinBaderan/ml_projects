import tensorflow as tf
from tensorflow.keras.datasets import mnist 
from tensorflow.keras.utils import to_categorical

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
    epochs=5,
    batch_size=100,
    validation_split=0.1
)

loss,accuracy = model.evaluate(x_test,y_test)

print(f"Accuracy : {accuracy}")

# predict 
prediction = model.predict(x_test)
print(prediction[0])
