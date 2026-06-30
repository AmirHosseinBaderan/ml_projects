import numpy as np
import matplotlib.pyplot as plt

from optimizer.optimizers.momentum import Momentum
from optimizer.optimizers.sgd import SGD

optimizer = SGD(learning_rate=0.1)

x = np.array([8.0])

history = []

for _ in range(20):

    history.append(x.copy())

    gradient = 2 * x

    x = optimizer.step(x, gradient)

x_values = np.linspace(-8,8,300)

y = x_values ** 2

plt.figure(figsize=(10,5))

plt.plot(x_values,y)

points = np.array(history)

plt.scatter(
    points,
    points**2,
    color="red"
)

plt.show()

optimizer = Momentum(
    learning_rate=0.1,
    momentum=0.9
)

x = np.array([8.0])

history = []

for _ in range(20):

    history.append(x.copy())

    gradient = 2*x

    x = optimizer.step(
        x,
        gradient
    )

x_values = np.linspace(-8,8,300)

y = x_values ** 2

plt.figure(figsize=(10,5))

plt.plot(x_values,y)

points = np.array(history)

plt.scatter(
    points,
    points**2,
    color="red"
)

plt.show()