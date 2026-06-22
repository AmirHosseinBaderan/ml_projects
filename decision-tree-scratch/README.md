# Decision Tree from Scratch

A Python implementation of a Decision Tree classifier built from scratch using only fundamental algorithms, without relying on machine learning libraries like scikit-learn for the core logic.

## What It Does

This project implements a Decision Tree classifier that predicts iris flower species based on their measurements (sepal length, sepal width, petal length, petal width). It uses the classic Iris dataset to train and evaluate the model's accuracy.

## Implementation

The implementation includes the following key components:

### Data Structures

- **`DecisionNode`**: Represents an internal node in the tree with:
  - `feature_idx`: Index of the feature used for splitting
  - `threshold`: Threshold value for the split
  - `left` and `right`: Child nodes

- **`LeafNode`**: Represents a terminal node with:
  - `value`: The predicted class label

### Core Algorithms

1. **Gini Impurity** (`gini` function):
   - Calculates the impurity of a set of labels
   - Formula: `1 - Σ(p_i²)` where p_i is the probability of class i

2. **Threshold Generation** (`get_threshold` function):
   - Generates potential split thresholds by finding midpoints between sorted unique values

3. **Dataset Splitting** (`split_dataset` function):
   - Divides the dataset into left and right subsets based on feature value and threshold

4. **Information Gain** (`information_gain` function):
   - Calculates the reduction in Gini impurity after a split
   - Used to determine the best split at each node

5. **Tree Building** (`build_tree` function):
   - Recursively builds the decision tree
   - Stops when all labels are the same or no beneficial split exists
   - Uses majority class for leaf nodes when no split is beneficial

6. **Prediction** (`predict` and `predict_many` functions):
   - Traverses the tree to make predictions
   - Supports batch prediction for multiple samples

7. **Accuracy Evaluation** (`accuracy` function):
   - Calculates classification accuracy on test data

### Training Process

1. Loads the Iris dataset from `data/iris.csv`
2. Splits data into training (80%) and test (20%) sets
3. Builds the decision tree using the training data
4. Makes predictions on the test set
5. Prints the tree structure and accuracy

## Usage

```bash
cd decision-tree-scratch
python main.py
```

## Dependencies

- `pandas`: For data loading and manipulation
- `scikit-learn`: For train/test split only (model is built from scratch)

## Example Output

The program outputs the tree structure showing the decision path and the final accuracy score on the test set.