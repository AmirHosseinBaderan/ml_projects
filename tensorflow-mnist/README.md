# TensorFlow MNIST

## What it does
A handwritten digit classifier built with TensorFlow/Keras. Trains a neural network on the MNIST dataset and provides an interactive viewer to inspect misclassified digits.

## How it works
1. Loads the MNIST dataset (28×28 grayscale images of digits 0–9)
2. Normalizes pixel values to [0, 1] by dividing by 255
3. One-hot encodes the labels for 10 classes
4. Builds a `Sequential` model with:
   - `Flatten` input layer (28×28 → 784)
   - Two `Dense` hidden layers (128 and 64 units, ReLU activation)
   - Output `Dense` layer (10 units, softmax activation)
5. Compiles with Adam optimizer and categorical crossentropy loss
6. Trains for 10 epochs with batch size 100 and 10% validation split
7. Evaluates on the test set and prints accuracy
8. Collects all misclassified images and displays them interactively:
   - Use left/right arrow keys to navigate through wrong predictions
   - Shows predicted label vs. true label for each misclassified digit

## Implementation
- `main.py`:
  - Loads data via `tensorflow.keras.datasets.mnist`
  - Preprocesses with `to_categorical` and normalization
  - Defines and trains the model
  - Evaluates and collects wrong predictions
  - Uses `matplotlib` for interactive image browsing with keyboard events

## Run
```bash
python main.py
```
