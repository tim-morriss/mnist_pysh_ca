"""Simple classification problem."""

import random
import os.path
import pandas as pd

from pyshgp.gp.genome import GeneSpawner
from pyshgp.push.instruction_set import InstructionSet
from pyshgp.gp.estimators import PushEstimator
from pyshgp.gp.selection import Lexicase


def dataset(filename):
    if os.path.isfile(filename):
        data = pd.read_csv(filename, sep=';', header=0)
        return data
    else:
        raise Exception('no file found')


def wine_quality(data_set):

    spawner = GeneSpawner(
        # Number of input instructions that could appear in the genomes.
        n_inputs=1,
        instruction_set=InstructionSet().register_core_by_stack({"bool", "int", "float"}),
        # A list of Literal objects to pull from when spawning genes and genomes.
        literals=list(range(11)),
        # A list of functions (aka Ephemeral Random Constant generators).
        # When one of these functions is called, the output is placed in a Literal and returned as the spawned gene.
        erc_generators=[lambda: random.randint(0, 11)]
    )

    selector = Lexicase(epsilon=False)

    estimator = PushEstimator(
        spawner=spawner,
        population_size=100,
        max_generations=50,
        selector=selector,
        verbose=2
    )

    X = data_set.iloc[:, :-1].head(100)
    y = data_set.iloc[:, -1:].head(100)
    print("X: %s \n y: %s" % (X.head(), y.head()))

    estimator.fit(X, y)


if __name__ == "__main__":
    file = '../../Data/wine_quality/winequality-white.csv'
    wine_quality(dataset(file))
