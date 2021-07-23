import random

import numpy as np

from load_datasets import LoadDatasets
from mnist_ca import MNISTCA, RunCA
from typing import Sequence
from pyshgp.gp.individual import Individual
from pyshgp.gp.genome import GeneSpawner
from pyshgp.push.instruction_set import InstructionSet
from pyshgp.gp.estimators import PushEstimator
# from pyshgp.gp.selection import Lexicase
from pyshgp.monitoring import VerbosityConfig
from pyshgp.gp.selection import Selector
from pyshgp.gp.population import Population
from pyshgp.tap import Tap, TapManager
from mnist_estimator import MNISTEstimator


def mnist_pysh_ca(pop_size=500, gens=100, steps=10, cut_size=None, digits=None):
    """
    Function to create and run the pyshGP + CA system

    @param pop_size: The size of the population in the GP algorithm
    @param gens: The amount of generations that the GP program will run for
    @param steps: The number of steps of CA update that will happen
    @param cut_size: The number of each digit of the MNIST dataset to include
    @param digits: The digits to include
    """
    spawner = GeneSpawner(
        # Number of input instructions that could appear in the genomes.
        n_inputs=1,
        # instruction_set="core",
        instruction_set=InstructionSet().register_core_by_stack({"bool", "int", "float"}),
        # A list of Literal objects to pull from when spawning genes and genomes.
        # literals=[np.float64(x) for x in digits],
        literals=[],
        # A list of functions (aka Ephemeral Random Constant generators).
        # When one of these functions is called, the output is placed in a Literal and returned as the spawned gene.
        erc_generators=[lambda: random.randint(0, 10)]
    )

    # selector = Lexicase(epsilon=False)

    verbosity = VerbosityConfig()

    estimator = MNISTEstimator(
        spawner=spawner,
        population_size=pop_size,
        max_generations=gens,
        selector=WackySelector(),
        variation_strategy="umad",
        last_str_from_stdout=True,
        verbose=2,
        steps=steps
    )

    X, y = LoadDatasets.load_mnist_tf()
    y = np.int64(y)     # for some reason pyshgp won't accept uint8, so cast to int64.
    X = X.reshape((-1, 784))
    y = y.reshape((-1, 1))
    X, y = LoadDatasets.exclusive_digits(X, y, digits, cut_size)

    # print("X: %s \n y: %s" % (X[0], y))

    TapManager.register("pyshgp.gp.search.SearchAlgorithm.step", MyCustomTap())
    TapManager.register("pyshgp.push.interpreter.PushInterpreter.run", StateTap())

    estimator.fit(X=X, y=y)

    best_solution = estimator.solution

    print("Program:\n", best_solution.program.code.pretty_str())
    # print("Test errors:\n", estimator.score(X, test_y_2d))


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


class StateTap(Tap):

    def pre(self, id, args, kwargs):
        interpreter = args[0]
        # print("Program: {0}".format(interpreter.program.pretty_str()))
        # print("Interpreter type library: {0}".format(interpreter.type_library))
        # print("Interpreter state: {0}".format(interpreter.state.pretty_print()))
        pass

# def fitness_function(individual, x, y, n):
#     average_intensities = {}
#     # intensity_by_class = {}
#     bin_label = []
#     for label in y:
#         average_intensities[label] = []
#     for (digit, label) in zip(x, y):
#         evolution = DrawCA(MNISTCA(digit, label, update_rule=individual.update_rule)).run(last_evolution_step=n)
#         average_intensities[label].append(evolution)
#     # for label in average_intensities.keys():
#     #     intensity_by_class[label] = np.average(np.array(average_intensities[label]).reshape(-1))
#     for label in average_intensities.keys():
#         bin_label.append(np.average(np.array(average_intensities[label]).reshape(-1)))
#     return abs(bin_label[0] - bin_label[1])

# class StateTap(Tap):

if __name__ == '__main__':
    mnist_pysh_ca(pop_size=300, gens=100, steps=10, cut_size=10, digits=[1, 2])
