# Iris Classification with KNN from Scratch

A Python implementation of the K-Nearest Neighbors (KNN) algorithm built from scratch to classify iris flower species based on their measurements.

## What It Does

This project implements a KNN classifier that predicts iris flower species (Setosa, Versicolor, Virginica) using the classic Iris dataset. It calculates distances between samples and uses the k-nearest neighbors to make predictions through majority voting.

## Implementation

The implementation consists of two files:

### `knn_classifier.py` - Core KNN Implementation

- **`euclidean_distance(x, y)`**: Calculates the Euclidean distance between two samples
  - Formula: `√(Σ(x_i - y_i)²)`

- **`KNNClassifier` class**:
  - `__init__(k=3)`: Initializes the classifier with k neighbors
  - `fit(x_train, y_train)`: Stores training data (KNN is a lazy learner)
  - `predict(sample)`: Predicts class for a single sample
    - Calculates distances to all training samples
    - Finds k nearest neighbors
    - Returns the majority class among neighbors
  - `predict_many(samples)`: Batch prediction for multiple samples

### `main.py` - Training and Evaluation

1. Loads the Iris dataset from `data/iris.csv`
2. Splits data into training (80%) and test (20%) sets
3. Creates a KNN classifier with k=15
4. Trains the model (stores training data)
5. Makes predictions on the test set
6. Calculates and prints accuracy

## How KNN Works

1. **Training**: Simply stores the training dataset (no actual training)
2. **Prediction**:
   - Calculate distance from the new sample to all training samples
   - Select the k closest samples (neighbors)
   - Vote among neighbors to determine the predicted class
3. **Majority Voting**: The class with the most votes among k neighbors is selected

## Usage

```bash
cd iris-from-scratch
python main.py
```

## Dependencies

- `pandas`: For data loading and manipulation
- `scikit-learn`: For train/test split only (KNN algorithm is built from scratch)
- `math`: For Euclidean distance calculation

## Example Output

The program outputs the classification accuracy on the test set.