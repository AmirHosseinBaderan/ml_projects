from neuron import Neuron

neuron = Neuron(
    weights=[0.4, 0.6],
    bias=1
)

result = neuron.forward([2, 5])

print(result)