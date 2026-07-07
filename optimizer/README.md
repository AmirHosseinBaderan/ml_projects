# Optimizer Implementations

A comprehensive collection of optimization algorithms implemented from scratch, including various optimizers, loss functions, regularizers, and visualization tools for understanding optimization dynamics.

## What It Does

This project implements and visualizes different optimization algorithms used in training neural networks. It provides:

- **Optimizers**: SGD, Momentum, RMSProp, Adam, and Batch Normalization
- **Loss Functions**: Mean Squared Error (MSE)
- **Regularizers**: L1 and L2 regularization
- **Test Functions**: Quadratic functions (1D and 2D) for optimization testing
- **Visualizations**: Loss curves, weight trajectories, and optimizer path plots

## Project Structure

```
optimizer/
├── main.py                           # Entry point - runs all optimizers
├── functions/
│   ├── base.py                       # Abstract base class for functions
│   ├── quadratic_1d.py               # 1D quadratic function: f(x) = x²
│   └── quadratic_2d.py               # 2D quadratic function: f(x,y) = x² + 10y²
├── optimizers/
│   ├── base.py                       # Abstract base class for optimizers
│   ├── sgd.py                        # Stochastic Gradient Descent
│   ├── momentum.py                   # Gradient descent with momentum
│   ├── rms_prop.py                   # RMSProp optimizer
│   ├── adam.py                       # Adam optimizer
│   ├── batch_norm.py                 # Batch Normalization implementation
│   └── dropout.py                    # Dropout regularization
├── losses/
│   └── mse.py                        # Mean Squared Error loss
├── regularizers/
│   ├── base.py                       # Abstract base class for regularizers
│   ├── l1.py                         # L1 regularization
│   └── l2.py                         # L2 regularization
├── runner/
│   ├── optimizer_runner.py           # Optimization runner
│   └── result.py                     # Result container
└── visualizers/
    ├── visualizer.py                 # Base visualizer (loss & weight plots)
    ├── function1d_visualizer.py      # 1D function path visualization
    ├── contour_visualizer.py           # 2D contour visualization
    └── animation.py                    # Animation support
```

## Implementation Details

### Optimizers

#### SGD (Stochastic Gradient Descent)
- **File**: [`optimizers/sgd.py`](optimizers/sgd.py)
- **Update Rule**: `weights = weights - learning_rate × gradients`
- **Parameters**: `learning_rate` (default: 0.1)

#### Momentum
- **File**: [`optimizers/momentum.py`](optimizers/momentum.py)
- **Update Rule**: `velocity = momentum × velocity - learning_rate × gradients`
- **Parameters**: `learning_rate` (default: 0.01), `momentum` (default: 0.9)
- **Purpose**: Accelerates convergence by accumulating past gradients

#### RMSProp
- **File**: [`optimizers/rms_prop.py`](optimizers/rms_prop.py)
- **Update Rule**: Maintains exponential moving average of squared gradients
- **Parameters**: `learning_rate` (default: 0.001), `beta` (default: 0.9), `epsilon` (default: 1e-8)
- **Purpose**: Adapts learning rate per parameter based on gradient magnitude

#### Adam
- **File**: [`optimizers/adam.py`](optimizers/adam.py)
- **Update Rule**: Combines momentum and RMSProp with bias correction
- **Parameters**: `learning_rate` (default: 0.001), `beta1` (default: 0.9), `beta2` (default: 0.999), `epsilon` (default: 1e-8)
- **Purpose**: Adaptive moment estimation - combines momentum and adaptive learning rates

#### BatchNorm
- **File**: [`optimizers/batch_norm.py`](optimizers/batch_norm.py)
- **Purpose**: Normalizes layer inputs for stable training
- **Parameters**: `num_features`, `momentum` (default: 0.9), `epsilon` (default: 1e-5)

#### Dropout
- **File**: [`optimizers/dropout.py`](optimizers/dropout.py)
- **Purpose**: Regularization technique to prevent overfitting
- **Parameters**: `p` (dropout probability, default: 0.5)

### Loss Functions

#### MSE (Mean Squared Error)
- **File**: [`losses/mse.py`](losses/mse.py)
- **Formula**: `loss = Σ(prediction - target)² / n`
- **Derivative**: `gradient = 2 × (prediction - target) / n`

### Regularizers

#### L1 Regularization
- **File**: [`regularizers/l1.py`](regularizers/l1.py)
- **Penalty**: `lambda × Σ|weights|`
- **Gradient**: `lambda × sign(weights)`

#### L2 Regularization
- **File**: [`regularizers/l2.py`](regularizers/l2.py)
- **Penalty**: `lambda × Σ(weights²)`
- **Gradient**: `2 × lambda × weights`

### Test Functions

#### Quadratic 1D
- **File**: [`functions/quadratic_1d.py`](functions/quadratic_1d.py)
- **Function**: `f(x) = x²`
- **Gradient**: `2x`

#### Quadratic 2D
- **File**: [`functions/quadratic_2d.py`](functions/quadratic_2d.py)
- **Function**: `f(x,y) = x² + 10y²`
- **Gradient**: `[2x, 20y]`

## How to Run

```bash
cd optimizer
python main.py
```

This runs all four optimizers (SGD, Momentum, RMSProp, Adam) on the 2D quadratic function starting from point `[8, 8]` with 20 iterations each.

## Visualizations

The project generates the following visualizations:

1. **Loss Plot**: Shows how the loss decreases over iterations
2. **Weight Plot**: Shows the trajectory of the first weight parameter
3. **1D Function Path**: For 1D functions, shows the optimizer path on the function curve
4. **Contour Plot**: For 2D functions, shows the optimizer path on a contour map

## Requirements

- Python 3.x
- numpy
- matplotlib

## Example Output

For each optimizer, the program displays:
- A loss curve showing convergence
- A weight trajectory plot
- For 2D functions, a contour plot showing the optimization path

The 2D quadratic function `x² + 10y²` has its minimum at `[0, 0]`, and all optimizers should converge toward this point.