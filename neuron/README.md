# Single Neuron Implementation

This project implements a **single artificial neuron** with forward propagation, demonstrating the fundamental building block of neural networks.

## What It Does

A neuron in a neural network computes a weighted sum of its inputs and adds a bias term. This implementation shows the basic mathematical operation:

```
z = (x₁ × w₁) + (x₂ × w₂) + ... + (xₙ × wₙ) + b
```

Where:
- `x` = input values
- `w` = weights
- `b` = bias

## Project Structure

```
neuron/
├── main.py       # Single neuron implementation and demo
└── README.md     # This file
```

## Implementation Details

### Neuron Function (`main.py`)

The [`neuron`](neuron/main.py:6) function implements the core neuron computation:

- **`inputs`** — List of input values to the neuron
- **`weights`** — List of weight values for each input
- **`bias`** — Bias term added to the weighted sum
- **Returns** — Tuple of `(z, weighted_sum)` where:
  - `weighted_sum` — Sum of element-wise products of inputs and weights
  - `z` — Final output after adding bias

## How to Run

```bash
cd neuron
python main.py
```

This runs a simple demonstration with:
- Inputs: `[10, 20, 30, 40]`
- Weights: `[0.1, 0.2, 0.3, -0.5]`
- Bias: `5`

## Dependencies

- Python 3.x
- NumPy (optional, for array operations)