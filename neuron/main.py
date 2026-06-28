x = [10, 20, 30, 40]
w = [0.1, 0.2, 0.3, -0.5]
b = 5


def neuron(inputs,weights,bias):
    weighted_sum = sum(
        x * w
        for x, w in zip(inputs, weights)
    )
    z = weighted_sum + bias

    return z,weighted_sum

z,weighted_sum = neuron(x, w, b)
print(f'Z : {z} / WS {weighted_sum}')