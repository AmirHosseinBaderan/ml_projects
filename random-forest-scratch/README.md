# Random Forest from Scratch

A Python implementation of Random Forest classifier that combines multiple decision trees using scikit-learn's `DecisionTreeClassifier` with custom ensemble logic.

## What It Does

This project implements a Random Forest classifier to predict iris flower species (Setosa, Versicolor, Virginica) based on their measurements. It uses an ensemble of decision trees with bootstrap sampling to improve classification accuracy and reduce overfitting.

## Implementation

The implementation consists of three files:

### `sample.py` - Bootstrap Sampling

- **`bootstrap_sample(X, y)`**: Creates random samples with replacement from the training data
  - Used to create diverse datasets for each tree in the forest
  - Returns sampled features and labels

### `forest.py` - Random Forest Class

- **`RandomForest` class**:
  - `__init__(n_trees=10, max_depth=None)`: Initializes the forest with specified number of trees
  - `fit(X, y)`: Trains the forest by:
    - Creating bootstrap samples for each tree
    - Training a `DecisionTreeClassifier` on each sample
  - `predict(X)`: Makes predictions by:
    - Getting predictions from all trees
    - Using majority voting to determine final class

### `main.py` - Training and Evaluation

1. Loads the Iris dataset from `data/iris.csv`
2. Splits data into training (80%) and test (20%) sets
3. Creates a Random Forest with 10 trees
4. Trains the model
5. Makes predictions on the test set
6. Calculates and prints accuracy

## How Random Forest Works

1. **Bootstrap Sampling**: Each tree is trained on a random subset of the training data (sampling with replacement)
2. **Ensemble Learning**: Multiple decision trees make independent predictions
3. **Majority Voting**: The final prediction is the class that receives the most votes from all trees

This approach reduces variance and overfitting compared to a single decision tree, as each tree sees a different subset of the data.

## Usage

```bash
cd random-forest-scratch
python main.py
```

## Dependencies

- `pandas`: For data loading and manipulation
- `numpy`: For numerical operations
- `scikit-learn`: For `DecisionTreeClassifier` and `train_test_split`
- `collections.Counter`: For majority voting

## Key Difference from `random-forest-sklearn`

While `random-forest-sklearn` uses scikit-learn's built-in `RandomForestClassifier`, this project implements the ensemble logic manually, giving more control over the training process and demonstrating how Random Forest works internally.