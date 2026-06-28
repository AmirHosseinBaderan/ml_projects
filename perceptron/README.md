# Perceptron Implementation

This project implements a **single-layer perceptron**, the simplest type of artificial neural network, demonstrating binary classification using a step activation function.

## What It Does

A perceptron computes a weighted sum of its inputs and applies a step function to produce a binary output (0 or 1). This is the fundamental building block for binary classification in neural networks.

```
z = (x₁ × w₁) + (x₂ × w₂) + ... + (xₙ × wₙ) + b
output = 1 if z ≥ 0, else 0
```

Where:
- `x` = input values
- `w` = weights
- `b` = bias

## Project Structure

```
perceptron/
├── main.py       # Perceptron implementation and demo
└── README.md     # This file
```

## Implementation Details

### Perceptron Function (`main.py`)

The [`perceptron`](perceptron/main.py:8) function implements the core perceptron computation:

- **`inputs`** — List of input values to the perceptron
- **`weights`** — List of weight values for each input
- **`bias`** — Bias term added to the weighted sum
- **Returns** — Binary output (1 if `z ≥ 0`, else 0)

The step activation function makes this perceptron suitable for simple binary classification tasks.

## How to Run

```bash
cd perceptron
python main.py
```

This runs a simple demonstration with:
- Inputs: `[80, 90, 18]`
- Weights: `[0.2, 0.3, 1]`
- Bias: `-40`

## Dependencies

- Python 3.x
- NumPy (optional, for array operations)