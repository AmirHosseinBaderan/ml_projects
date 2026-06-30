from functions.quadratic_2d import Quadratic2D
from optimizers.sgd import SGD
from optimizer.runner.optimizer_runner import OptimizerRunner
from optimizers.momentum import Momentum

from visualizers.visualizer import Visualizer
from visualizers.function1d_visualizer import Function1DVisualizer

def run(
    optimizer,
    function,
    start,
    iterations=20,
):
    runner = OptimizerRunner(
        optimizer=optimizer,
        function=function,
    )

    result = runner.run(
        start=start,
        iterations=iterations,
    )

    Visualizer().plot_loss(result)
    Visualizer().plot_weight(result)

    if len(start) == 1:
        Function1DVisualizer().plot(function, result)

run(
    optimizer=SGD(0.1),
    function=Quadratic2D(),
    start=[8, 8],
)
run(
    Momentum(
        learning_rate=0.1,
        momentum=0.9
    ),
    function=Quadratic2D(),
    start=[8, 8],
)