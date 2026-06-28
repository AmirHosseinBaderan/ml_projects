def neuron(input, weight, bias):
    weighted_sum = 0

    for x, w in zip(input, weight):
        weighted_sum += x * w

    z = weighted_sum + bias
    return activation(z)


def activation(z):
    if z >= 0:
        return z

    return 0


inputs = [2, 5]


def forward(inputs):
    hidden_weights = [
        [0.4, 0.6],
        [-0.2, 0.8]
    ]

    hidden_bias = [
        1,
        -1
    ]

    hidden_outputs = []

    for weight, bias in zip(hidden_weights, hidden_bias):
        hidden_outputs.append(
            neuron(
                inputs,
                weight,
                bias
            )
        )

    output = neuron(
        hidden_outputs,
        [0.5, 0.3],
        -2
    )

    return hidden_outputs,output


prediction = forward(inputs)
print(prediction)
