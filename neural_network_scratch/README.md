# Neural Network from Scratch

This project implements a **complete neural network library** from scratch, including forward propagation, backpropagation, and various activation functions and optimizers.

## What It Does

A modular neural network implementation that supports:
- Multiple layer architectures
- Various activation functions (Sigmoid, ReLU, Tanh, LeakyReLU)
- Mean Squared Error loss
- Stochastic Gradient Descent optimizer
- Training on XOR-like dataset

## Project Structure

```
neural_network_scratch/
├── main.py           # Demo training XOR-like problem
├── network.py        # Network class for layer management
├── layer.py          # Layer class with multiple neurons
├── neuron.py         # Individual neuron with forward/backward
├── activations.py    # Activation functions (Sigmoid, ReLU, etc.)
├── losses.py         # Loss functions (MSE, BinaryCrossEntropy)
├── optimizers.py     # SGD optimizer
├── module.py         # Base module class
├── initializer.py    # Weight initialization strategies
└── SGD_plt.png       # Training visualization
```

## Implementation Details

### Network (`network.py`)

The [`Network`](neural_network_scratch/network.py:5) class manages the network architecture:
- **`add(layer)`** — Adds a layer to the network
- **`forward(inputs)`** — Performs forward propagation through all layers
- **`fit(x_train, y_train, optimizer, loss, epochs, verbose)`** — Trains the network using backpropagation
- **`predict(inputs)`** — Makes predictions (forward pass only)
- **`backward(loss)`** — Performs backward propagation through layers

The training loop in `fit()`:
1. For each epoch, iterates through all training samples
2. Performs forward pass to get predictions
3. Computes loss value
4. Calls `backward()` to propagate error gradients
5. Updates weights using the optimizer
6. Prints epoch loss if verbose

### Layer (`layer.py`)

The [`Layer`](neural_network_scratch/layer.py:5) class represents a layer of neurons:
- **`forward(inputs)`** — Forward pass through all neurons, returns list of outputs
- **`backward(deltas)`** — Backward pass, computes and accumulates gradients

Each layer maintains:
- `neurons` — List of Neuron instances
- `inputs` / `outputs` — Cached values for backpropagation
- `deltas` — Gradient values for weight updates

### Neuron (`neuron.py`)

The [`Neuron`](neural_network_scratch/neuron.py:7) class implements individual neurons with:
- **`forward(inputs)`** — Computes weighted sum `z = Σ(x_i × w_i) + bias`, applies activation function
- **`backward(delta)`** — Computes gradients:
  - `weight_gradients = [delta × x_i for each input]`
  - `bias_gradient = delta`
  - Returns `previous_deltas = [delta × w_i for each weight]`

The neuron stores:
- `weights` / `bias` — Learnable parameters
- `inputs` / `z` / `output` — Cached values for backpropagation
- `delta` / `weight_gradients` / `bias_gradient` — Gradient values

### Activations (`activations.py`)

Available activation functions:
- **`Sigmoid`** — `1 / (1 + e^(-x))`, derivative: `output × (1 - output)`
- **`ReLU`** — `max(0, x)`, derivative: `1 if output > 0 else 0`
- **`Tanh`** — `tanh(x)`, derivative: `1 - output²`
- **`LeakyReLU`** — `x if x > 0 else alpha × x`, derivative: `1 if output > 0 else alpha`
- **`Identity`** — `x` (no activation), derivative: `1`

### Losses (`losses.py`)

- **`MSE`** — Mean Squared Error: `loss = Σ(p - t)² / n`, derivative: `2 × (p - t) / n`
- **`BinaryCrossEntropy`** — Placeholder for future implementation

### Optimizers (`optimizers.py`)

- **`SGD`** — Stochastic Gradient Descent:
  - Updates weights: `w_i = w_i - learning_rate × gradient`
  - Updates bias: `bias = bias - learning_rate × bias_gradient`

### Initializer (`initializer.py`)

- **`RandomUniformInitializer`** — Random weights in range `[-1, 1]`
- **`XavierInitializer`** — Placeholder (for sigmoid/tanh)
- **`HeInitializer`** — Placeholder (for ReLU)

## Algorithm Overview

### Forward Propagation
1. Input passes through each layer sequentially
2. Each neuron computes: `z = weighted_sum + bias`
3. Activation function applied: `output = activation(z)`
4. Layer outputs become next layer inputs

### Backward Propagation
1. Loss derivative computed: `dL/doutput`
2. For each layer (in reverse):
   - Activation derivative applied: `delta = dL/doutput × activation_derivative`
   - Weight gradients: `dL/dw = delta × input`
   - Bias gradient: `dL/db = delta`
   - Previous deltas: `dL/dprev = delta × weight`
3. Optimizer updates weights and biases

## How to Run

```bash
cd neural_network_scratch
python main.py
```

This trains a 2-layer network on an XOR-like dataset:
- Input: 2 features
- Hidden layer: 2 neurons with Sigmoid activation
- Output layer: 1 neuron with Sigmoid activation
- Training: 5000 epochs with learning rate 0.5

## Dependencies

- Python 3.x
- NumPy (optional, for array operations)
- Matplotlib (for visualization)