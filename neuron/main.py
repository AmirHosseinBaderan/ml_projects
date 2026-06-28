inputs = [5, 8, 2]
weights = [0.5, -0.4, 2]

weighted_sum = 0
bias = 3

for i in range(len(inputs)):
    weighted_sum += inputs[i] * weights[i]

z = weighted_sum + bias

print(z)
