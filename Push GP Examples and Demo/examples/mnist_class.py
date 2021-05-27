import random
import numpy as np
# For loading MNIST
from struct import unpack
from typing import Sequence
from pyshgp.gp.individual import Individual
from pyshgp.gp.genome import GeneSpawner
from pyshgp.push.instruction_set import InstructionSet
from pyshgp.gp.estimators import PushEstimator
# from pyshgp.gp.selection import Lexicase
from pyshgp.gp.selection import Selector
from pyshgp.gp.population import Population
from pyshgp.tap import Tap, TapManager


def load_mnist(image_file, label_file, cut_size=0):
    # Open image and label files in binary mode
    images = open(image_file, 'rb')
    labels = open(label_file, 'rb')

    # Get metadata for images
    images.read(4)  # Skip the magic_number
    number_of_images = images.read(4)
    number_of_images = unpack('>I', number_of_images)[0]
    rows = images.read(4)
    rows = unpack('>I', rows)[0]
    cols = images.read(4)
    cols = unpack('>I', cols)[0]

    # Get metadata for labels
    labels.read(4)
    N = labels.read(4)
    N = unpack('>I', N)[0]

    # Get data
    X = np.zeros((N, rows * cols), dtype=np.int64)  # Initialise X
    y = np.zeros(N, dtype=np.int64)  # Initialise y
    for i in range(N):
        for j in range(rows * cols):
            tmp_pixel = images.read(1)
            tmp_pixel = unpack('>B', tmp_pixel)[0]
            X[i][j] = tmp_pixel
        tmp_label = labels.read(1)
        y[i] = unpack('>B', tmp_label)[0]

    images.close()
    labels.close()

    print("MNIST Loaded with a cut size of: %i" % cut_size)
    if cut_size == 0:
        return X, y
    else:
        return X[:cut_size], y[:cut_size]


def mnist_pysh(train_x, train_y, test_x, test_y, pop_size=500, gens=100):
    spawner = GeneSpawner(
        # Number of input instructions that could appear in the genomes.
        n_inputs=1,
        instruction_set=InstructionSet().register_core_by_stack({"int"}),
        # A list of Literal objects to pull from when spawning genes and genomes.
        literals=list(range(11)),
        # A list of functions (aka Ephemeral Random Constant generators).
        # When one of these functions is called, the output is placed in a Literal and returned as the spawned gene.
        erc_generators=[lambda: random.randint(0, 10)]
    )

    # selector = Lexicase(epsilon=False)

    estimator = PushEstimator(
        spawner=spawner,
        population_size=pop_size,
        max_generations=gens,
        selector=WackySelector(),
        variation_strategy="umad",
        last_str_from_stdout=True,
        verbose=2
    )

    X = train_x
    y = [[label] for label in train_y]
    test_y_2d = [[label] for label in test_y]
    print("X: %s \n y: %s" % (X[0], y))

    TapManager.register("pyshgp.gp.search.SearchAlgorithm.step", MyCustomTap())

    estimator.fit(X=X, y=y)

    best_solution = estimator.solution

    print("Program:\n", best_solution.program.code.pretty_str())
    print("Test errors:\n", estimator.score(test_x, test_y_2d))


class WackySelector(Selector):
    """
    A parent selection algorithm that gets progressively more elitist with each selection.

    The first time the `WackySelector` is used, it will select a random individual.
    Every subsequent selection will select a parent with a total error that is <= the previous.
    If there are no individuals in the population with a <= total error, a random individual will be selected.
    """

    def __init__(self):
        self.threshold = None

    def select_random(self, pool: Sequence[Individual]) -> Individual:
        selected = random.choice(pool)
        self.threshold = selected.total_error
        return selected

    def select_one(self, population: Population) -> Individual:
        """Return single individual from population."""
        if self.threshold is None:
            return self.select_random(population)
        else:
            filtered = [ind for ind in population if ind.total_error <= self.threshold]
            if len(filtered) == 0:
                return self.select_random(population)
            else:
                return self.select_random(filtered)

    def select(self, population: Population, n: int = 1) -> Sequence[Individual]:
        """Return `n` individuals from the population."""
        super().select(population, n)
        return [self.select_one(population) for _ in range(n)]


class MyCustomTap(Tap):

    def pre(self, id: str, args, kwargs, obj=None):
        """Print population stats before the next step of the run."""
        search = args[0]
        best_individual = search.population.best()
        print()
        print("Generation:", search.generation)
        print("Best Program:", best_individual.program.pretty_str())
        print("Best Error Vector:", best_individual.error_vector)
        print("Best Total Error:", best_individual.total_error)


if __name__ == '__main__':
    train_X, train_y = load_mnist(
        '../../Data/mnist/train-images.idx3-ubyte',
        '../../Data/mnist/train-labels.idx1-ubyte',
        1000
    )

    test_X, test_y = load_mnist(
        '../../Data/mnist/t10k-images.idx3-ubyte',
        '../../Data/mnist/t10k-labels.idx1-ubyte',
        1000
    )

    mnist_pysh(train_X, train_y, test_X, test_y, 300, 10)
