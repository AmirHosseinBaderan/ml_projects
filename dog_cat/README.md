# Dog vs Cat Classifier

## What it does
A binary image classifier that distinguishes between cats and dogs using a Convolutional Neural Network (CNN) built with TensorFlow/Keras. Supports training on a dataset and predicting on single images or entire folders.

## How it works
1. Loads images from `data/training_set` and `data/test_set` directories using `ImageDataGenerator`
2. Applies data augmentation (shear, zoom, horizontal flip) to the training set and rescales all images to [0, 1]
3. Builds a CNN model with the following architecture:
   - **Input**: 150×150 RGB images (3 channels)
   - **Conv2D (32 filters, 3×3 kernel, ReLU)**: Extracts low-level features like edges and textures
   - **MaxPooling2D (2×2)**: Reduces spatial dimensions by half, providing translation invariance
   - **Conv2D (64 filters, 3×3 kernel, ReLU)**: Learns mid-level features like patterns and shapes
   - **MaxPooling2D (2×2)**: Further downsampling
   - **Conv2D (128 filters, 3×3 kernel, ReLU)**: Learns high-level features like object parts
   - **MaxPooling2D (2×2)**: Final downsampling before fully connected layers
   - **Flatten**: Converts 3D feature maps into a 1D vector
   - **Dense (512 units, ReLU)**: Fully connected layer for high-level reasoning
   - **Dense (1 unit, sigmoid)**: Output layer producing a probability between 0 (cat) and 1 (dog)
4. Compiles with Adam optimizer and binary crossentropy loss
5. Trains for 20 epochs if no saved model exists, otherwise loads the existing `cat_dog_model.keras`
6. Predicts classes using a 0.5 threshold on the sigmoid output
7. Supports both single-image and folder-level prediction via command-line arguments

## Implementation
- `main.py`:
  - Sets up data generators with augmentation and preprocessing
  - Defines `create_model()` — builds the CNN architecture
  - Loads existing model or trains and saves a new one
  - Uses `Predictor` class to run inference on images or folders
- `predictor.py`:
  - `Predictor` class wraps the model for inference
  - `predict_image(path)` — reads an image with OpenCV, resizes to 150×150, normalizes, and returns the predicted class
  - `predict_folder(path)` — iterates over all files in a folder and counts cat/dog predictions
- `model_plt.py`:
  - `show_model_plot(history)` — plots training/validation accuracy and loss curves using matplotlib

## Run
```bash
# Train the model (if not already saved)
python main.py

# Predict on a single image
python main.py path/to/image.jpg

# Predict on all images in a folder
python main.py path/to/folder/
```
