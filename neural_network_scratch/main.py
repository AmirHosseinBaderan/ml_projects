from network import Network
from layer import Layer
from activations import Sigmoid
from losses import MSE
from optimizers import SGD

X = [
    [0, 0],
    [0, 1],
    [1, 0],
    [1, 1]
]

y = [
    [0],
    [0],
    [0],
    [1]
]

network = Network()

network.add(
    Layer(
        neuron_count=2,
        input_size=2,
        activation=Sigmoid()
    )
)

network.add(
    Layer(
        neuron_count=1,
        input_size=2,
        activation=Sigmoid()
    )
)

optimizer = SGD(
    learning_rate=0.5
)

loss = MSE()

network.fit(
    X,
    y,
    optimizer=optimizer,
    loss=loss,
    epochs=5000
)

for sample in X:
    print(
        sample,
        network.predict(sample)
    )

# learning rate
SGD(learning_rate=0.001)

SGD(learning_rate=0.01)

SGD(learning_rate=0.5)

SGD(learning_rate=5)