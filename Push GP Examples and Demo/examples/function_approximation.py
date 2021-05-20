"""A example of function approximation."""

import numpy as np
import random

from pyshgp.gp.selection import Lexicase
from pyshgp.gp.selection import Tournament
from pyshgp.gp.estimators import PushEstimator
from pyshgp.gp.genome import GeneSpawner
from pyshgp.push.instruction_set import InstructionSet
from pyshgp.validation import check_X_y


def target_function(a):
    return a * a


def function_approximation(func):
    # Needs to be reshaped in order for it to work
    X = np.arange(50).reshape(-1, 2)
    y = np.array([func(x) for x in X])
    print("x: %s \n y: %s" % (X, y))
    # Checks to see if the X and y values are valid.
    # Throws error if not.
    print(check_X_y(X, y))

    instruction_set = (
        # Create a int stack but exclude str, exec, and code
        InstructionSet().register_core_by_stack({"int"}, exclude_stacks={"str", "exec", "code"})
    )

    spawner = GeneSpawner(
        # Number of input instructions that could appear in the genomes.
        n_inputs=1,
        instruction_set=instruction_set,
        # A list of Literal objects to pull from when spawning genes and genomes.
        literals=[],
        # A list of functions (aka Ephemeral Random Constant generators).
        # When one of these functions is called, the output is placed in a Literal and returned as the spawned gene.
        erc_generators=[lambda: random.randint(0, 10)]
    )

    selector = Tournament()

    est = PushEstimator(
        population_size=300,
        max_generations=30,
        simplification_steps=500,
        spawner=spawner,
        selector=selector,
        verbose=2
    )

    est.fit(X=X, y=y)


if __name__ == '__main__':
    function_approximation(target_function)
