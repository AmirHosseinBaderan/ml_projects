# main.py

 
# CNN Architectures Evolution
# Educational Version
 


class Conv:
    def __init__(self, filters, kernel):
        self.filters = filters
        self.kernel = kernel

    def __repr__(self):
        return f"Conv(filters={self.filters}, kernel={self.kernel})"


class AveragePool:
    def __repr__(self):
        return "AveragePool()"


class MaxPool:
    def __repr__(self):
        return "MaxPool()"


class ReLU:
    def __repr__(self):
        return "ReLU()"


class Flatten:
    def __repr__(self):
        return "Flatten()"


class Dense:
    def __init__(self, units):
        self.units = units

    def __repr__(self):
        return f"Dense({self.units})"


class Softmax:
    def __repr__(self):
        return "Softmax()"


class Dropout:
    def __init__(self, rate):
        self.rate = rate

    def __repr__(self):
        return f"Dropout({self.rate})"


class ResidualBlock:
    def __init__(self, filters):
        self.filters = filters

    def __repr__(self):
        return f"ResidualBlock(filters={self.filters})"


class MBConv:
    def __init__(self, filters):
        self.filters = filters

    def __repr__(self):
        return f"MBConv(filters={self.filters})"


# LeNet (1998)
LeNet = [
    Conv(6, 5),
    AveragePool(),

    Conv(16, 5),
    AveragePool(),

    Flatten(),

    Dense(120),
    Dense(84),
    Dense(10),

    Softmax()
]

# AlexNet (2012)
AlexNet = [
    Conv(96, 11),
    ReLU(),
    MaxPool(),

    Conv(256, 5),
    ReLU(),
    MaxPool(),

    Conv(384, 3),
    ReLU(),

    Conv(384, 3),
    ReLU(),

    Conv(256, 3),
    ReLU(),
    MaxPool(),

    Flatten(),

    Dense(4096),
    ReLU(),
    Dropout(0.5),

    Dense(4096),
    ReLU(),
    Dropout(0.5),

    Dense(1000),
    Softmax()
]

# VGG16 (2014)
VGG16 = [

    Conv(64, 3),
    ReLU(),
    Conv(64, 3),
    ReLU(),
    MaxPool(),

    Conv(128, 3),
    ReLU(),
    Conv(128, 3),
    ReLU(),
    MaxPool(),

    Conv(256, 3),
    ReLU(),
    Conv(256, 3),
    ReLU(),
    Conv(256, 3),
    ReLU(),
    MaxPool(),

    Conv(512, 3),
    ReLU(),
    Conv(512, 3),
    ReLU(),
    Conv(512, 3),
    ReLU(),
    MaxPool(),

    Conv(512, 3),
    ReLU(),
    Conv(512, 3),
    ReLU(),
    Conv(512, 3),
    ReLU(),
    MaxPool(),

    Flatten(),

    Dense(4096),
    ReLU(),

    Dense(4096),
    ReLU(),

    Dense(1000),
    Softmax()
]

# ResNet50 (Concept)
ResNet = [

    Conv(64, 7),
    ReLU(),
    MaxPool(),

    ResidualBlock(64),
    ResidualBlock(64),
    ResidualBlock(64),

    ResidualBlock(128),
    ResidualBlock(128),
    ResidualBlock(128),
    ResidualBlock(128),

    ResidualBlock(256),
    ResidualBlock(256),
    ResidualBlock(256),
    ResidualBlock(256),
    ResidualBlock(256),
    ResidualBlock(256),

    ResidualBlock(512),
    ResidualBlock(512),
    ResidualBlock(512),

    Flatten(),

    Dense(1000),
    Softmax()
]

# EfficientNet (Concept)
EfficientNet = [

    Conv(32, 3),

    MBConv(16),

    MBConv(24),
    MBConv(24),

    MBConv(40),
    MBConv(40),

    MBConv(80),
    MBConv(80),
    MBConv(80),

    MBConv(112),
    MBConv(112),
    MBConv(112),

    MBConv(192),
    MBConv(192),
    MBConv(192),
    MBConv(192),

    MBConv(320),

    Flatten(),

    Dense(1000),

    Softmax()
]

architectures = {
    "LeNet": LeNet,
    "AlexNet": AlexNet,
    "VGG16": VGG16,
    "ResNet50": ResNet,
    "EfficientNet": EfficientNet,
}

for name, layers in architectures.items():
    print("=" * 60)
    print(name)
    print("=" * 60)

    for i, layer in enumerate(layers, start=1):
        print(f"{i:02d}. {layer}")

    print()