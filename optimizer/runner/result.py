class OptimizationResult:

    def __init__(self):
        self.weight_history = []
        self.gradients_history = []
        self.losses_history = []
        self.velocity_history = []