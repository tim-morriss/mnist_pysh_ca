from typing import Sequence
import random

import numpy as np
from pyshgp.gp.estimators import PushEstimator
from pyshgp.gp.genome import GeneSpawner

from pyshgp.gp.individual import Individual
from pyshgp.gp.population import Population
from pyshgp.gp.selection import Selector
from pyshgp.push.instruction_set import InstructionSet
from pyshgp.tap import tap, Tap, TapManager

"""
For this demo, we will attempt to evolve a program that solved a simplified 
version of `FizzBuzz`. The problem prompt is as follows:

    Given an integer, print "Fizz" if it is divisible by 3, print 
    "Buzz" if it is divisible by 5, and print "FizzBuzz" if it is divisible 
    "Buzz" if it is divisible by 5, and print "FizzBuzz" if it is divisible 
    by both 3 and 5.
    
To start, we will create two datasets of example input-output pairs.
One will be used for training and the other will be used to test our program on unseen data.

To get the true labels of our datasets, we will manually write a `fizz_buzz` program.
"""


def fizz_buzz(i: int) -> str:
    s = ""
    if i % 3 == 0:
        s += "Fizz"
    if i % 5 == 0:
        s += "Buzz"
    return s


x_train = np.arange(30).reshape(-1, 1)  # Use the numbers 0 to 30 to train
y_train = np.array([fizz_buzz(x[0]) for x in x_train]).reshape(-1, 1)

x_test = np.arange(30, 50).reshape(-1, 1)  # Use the numbers 30 to 50 to test
y_test = np.array([fizz_buzz(x[0]) for x in x_train]).reshape(-1, 1)


"""
When configuring evolution, we can use the selection algorithms and genetic
operators that are built into `pyshgp` or we can define our own.

For the purposes of demonstration, we will define a custom parent selection algorithm.
"""


class WackySelector(Selector):
    """A parent selection algorithm that gets progressively more elitist with each selection.

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


"""
Next we have to declare our `GeneSpawner`. 

A spawner holds all the instructions, literals, erc generators, and inputs 
that we want to appear in our genomes/programs. It will be used by evolution
to generate random genomes for our initial population and random genes for mutation.
"""

spawner = GeneSpawner(
    n_inputs=1,
    instruction_set=InstructionSet().register_core_by_stack({"str", "int"}),
    literals=["Fizz", "Buzz"],
    erc_generators=[
        lambda: random.randint(0, 10),
    ]
)


"""
We now have everything we need to configure a run of PushGP.

We will create a `PushEstimator` and parameterize it however we want. Let's be sure to 
pass an instance of our custom parent selector.
"""


est = PushEstimator(
    spawner=spawner,
    selector=WackySelector(),
    variation_strategy="umad",
    last_str_from_stdout=True,
    population_size=300,
    max_generations=100,
    initial_genome_size=(10, 50),
    simplification_steps=2000,
    parallelism=False,
    verbose=2
)


"""
Before we start our PushGP, let's customize what will be printed to stdout each generation.

By default, `pyshgp` will print a lot of statistics each generation of evolution. Perhaps we
want to show less.

The `TapManager` provided by `pyshgp` will run a side effect (like printing or logging) before
and after a certain method is called. 

Let's define a custom `Tap` that will print before each generation. We specify the tap is for 
`step` method of all `SearchAlgorithm` objects using the function id `pyshgp.gp.search.SearchAlgorithm.step`
and the specify the tap should print before by implementing the `pre` method of `Tap`.
"""


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


TapManager.register("pyshgp.gp.search.SearchAlgorithm.step", MyCustomTap())


"""
Now let's kick off our PushGP run!

We will call the estimator's `.fit` method on our training data, and then call `.score` on our 
test data. If the test errors are all zero, we found a generalizing solution!
"""

if __name__ == "__main__":
    est.fit(X=x_train, y=y_train)
    best_found = est.solution

    print("Program:\n", best_found.program.code.pretty_str())
    print("Test errors:\n", est.score(x_test, y_test))
