x = [5, 8, 2]
w = [0.5, -0.4, 2]
b = 3


def neuron(inputs,weights,bias):
    weighted_sum = 0

    for i in range(len(inputs)):
        weighted_sum += inputs[i] * weights[i]

    z = weighted_sum + bias
    return z

output = neuron(x, w, b)
print(output)