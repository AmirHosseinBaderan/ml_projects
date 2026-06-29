# Multi-Layer Perceptron (MLP)

This project implements a **multi-layer perceptron** with one hidden layer, demonstrating forward propagation through a simple neural network architecture.

## What It Does

A multi-layer perceptron consists of multiple neurons organized in layers. This implementation shows:
- A hidden layer with 2 neurons
- An output layer with 1 neuron
- ReLU activation function for the hidden layer

## Project Structure

```
mlp/
├── main.py       # MLP implementation and demo
└── README.md     # This file
```

## Implementation Details

### Neuron Function (`main.py`)

The [`neuron`](mlp/main.py:1) function computes the weighted sum and applies activation:

- **`input`** — List of input values
- **`weight`** — List of weight values
- **`bias`** — Bias term
- **Returns** — Activated output (ReLU)

The computation:
```
z = Σ(x_i × w_i) + bias
output = activation(z)
```

### Activation Function

The [`activation`](mlp/main.py:11) function implements ReLU (Rectified Linear Unit):
- Returns `z` if `z >= 0`
- Returns `0` otherwise

### Forward Function

The [`forward`](mlp/main.py:21) function performs forward propagation:
1. Computes hidden layer outputs using 2 neurons with ReLU activation
2. Passes hidden outputs to the output neuron (no activation)
3. Returns both hidden and final outputs

## How to Run

```bash
cd mlp
python main.py
```

This runs a demonstration with:
- Inputs: `[2, 5]`
- Hidden weights: `[[0.4, 0.6], [-0.2, 0.8]]`
- Hidden biases: `[1, -1]`
- Output weights: `[0.5, 0.3]`
- Output bias: `-2`

## Dependencies

- Python 3.x
- NumPy (optional, for array operations)