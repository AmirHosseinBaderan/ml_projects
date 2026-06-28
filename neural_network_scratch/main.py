from network import Network
from  layer import Layer

network = Network()

# add layers
network.add(
    Layer(
        neuron_count=4,
        input_size=2
    )
)

network.add(
    Layer(
        neuron_count=2,
        input_size=4
    )
)

network.add(
    Layer(
        neuron_count=1,
        input_size=2
    )
)

result = network.predict([1000, -500])
print(result)