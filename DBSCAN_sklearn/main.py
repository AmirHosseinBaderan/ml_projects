import matplotlib.pyplot as plt

X = [
    [1, 2],
    [2, 2],
    [2, 3],
    [8, 8],
    [8, 9],
    [25, 25]
]

x = [p[0] for p in X]
y = [p[1] for p in X]

plt.scatter(x,y)
plt.grid(True)
plt.show()