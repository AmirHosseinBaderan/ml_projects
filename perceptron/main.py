inputs = [80, 90, 18]

weights = [0.2, 0.3, 1]

bias = -40


def perceptron(inputs, weights, bias):
    weighted_sum = 0

    for x, w in zip(inputs, weights):
        weighted_sum += x * w

    z = weighted_sum + bias

    if z >= 0:
        return 1

    return 0

result = perceptron(inputs, weights, bias)

print(result)