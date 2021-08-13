import random
import numpy as np
import argparse
import re

from typing import Sequence, List, Set
from load_datasets import LoadDatasets
from pyshgp.gp.individual import Individual
from pyshgp.gp.genome import GeneSpawner
from pyshgp.push.instruction_set import InstructionSet
from pyshgp.monitoring import VerbosityConfig
from pyshgp.gp.selection import Selector, Lexicase
from pyshgp.gp.population import Population
from pyshgp.tap import Tap, TapManager
from mnist_estimator import MNISTEstimator
from ca_animate import CAAnimate


def mnist_pysh_ca(
        mode: str,
        load_filepath: str,
        save_filepath: str,
        pop_size: int = 500,
        gens: int = 100,
        steps: int = 10,
        cut_size: int = None,
        digits: List = None,
        shuffle: bool = False,
        simplifcation: int = 2000,
        stacks: Set[str] = None,
        picture: bool = False):
    """
    Function to create and run the pyshGP + CA system

    Parameters
    ----------
    mode: Mode
        Choose between training and testing operation modes
    load_filepath: str
        Filepath used for loading existing models for testing and simplification.
    save_filepath: str
        Filepath used for saving the estimator during training runs
    pop_size: int
        The size of the population in the GP algorithm
    gens: int
        The amount of generations that the GP program will run for
    steps: int
        The number of steps of CA update that will happen
    cut_size: int
        The number of each digit of the MNIST dataset to include
    digits: List
        The digits to include
    shuffle: bool
        Whether or not to shuffle the selection from the MNIST data set
    simplifcation: int
        Number of simplification steps to do.
    stacks: str
        Which stacks to use when running the program
    """

    modes = ['training', 'testing', 'simplify']

    if mode.lower() not in modes:
        raise ValueError("Invalid mode. Expected either {0}, {1}, or {2}.".format(modes[0], modes[1], modes[2]))

    if next(iter(stacks)) == "core":
        stack_parse = "core"
    else:
        if stacks is None:
            stack_parse = InstructionSet().register_core_by_stack({"float"})
        else:
            stack_parse = InstructionSet().register_core_by_stack(stacks)

    spawner = GeneSpawner(
        # Number of input instructions that could appear in the genomes.
        n_inputs=1,
        # instruction_set="core",
        instruction_set=stack_parse,
        # A list of Literal objects to pull from when spawning genes and genomes.
        literals=[np.float64(x) for x in np.arange(digits[0], digits[-1], 0.1)],
        # A list of functions (aka Ephemeral Random Constant generators).
        # When one of these functions is called, the output is placed in a Literal and returned as the spawned gene.
        erc_generators=[lambda: random.randint(0, 10)]
    )

    selector = Lexicase(epsilon=False)

    verbosity = VerbosityConfig()

    estimator = MNISTEstimator(
        spawner=spawner,
        population_size=pop_size,
        max_generations=gens,
        selector=selector,
        variation_strategy="umad",
        last_str_from_stdout=True,
        verbose=2,
        steps=steps,
        simplification_steps=0
    )

    X, y = LoadDatasets.load_mnist_tf()
    y = np.int64(y)  # for some reason pyshgp won't accept uint8, so cast to int64.
    X = X.reshape((-1, 784))
    y = y.reshape((-1, 1))
    X, y = LoadDatasets.exclusive_digits(X, y, digits, cut_size, shuffle)

    if mode == 'testing':
        estimator.load(load_filepath)
        score = estimator.score(X=X, y=y)
        print("\n Error vector: \n", score)
        print("Total Error: \n", score.sum())
        output = estimator.evaluator.output
        print("CA output: \n", output)

        def filepath(ending, extension):
            return 'output/{0}--digit{1}-{2}steps-cut_size{3}{4}.{5}'.format(
                load_filepath.split('.')[0],
                digits[1], steps, cut_size, ending, extension
            )

        if picture:
            # print("Grid output: \n", estimator.evaluator.error_function.last_ca_grid.shape)
            CAAnimate.animate_ca(
                estimator.evaluator.error_function.last_ca_grid,
                filepath("", "gif")
            )
        with open(filepath("test", "txt"), 'w') as f:
            f.write("Error vector: \n {0}".format(score))
            f.write("Total Error: \n {0}".format(score.sum()))
            f.write("CA output: \n {0}".format(output))

    if mode == 'training':
        TapManager.register("pyshgp.gp.search.SearchAlgorithm.step", MyCustomTap())
        estimator.fit(X=X, y=y)
        estimator.save(save_filepath)
        best_solution = estimator.solution
        print("Program:\n", best_solution.program.code.pretty_str())
        if simplifcation > 0:
            estimator.simplify(X=X, y=y, simplification_steps=200)
            estimator.save(save_filepath)
            print("Program simplified:\n", best_solution.program.code.pretty_str())

    if mode == 'simplify':
        estimator.load(load_filepath)
        estimator.simplify(X=X, y=y, simplification_steps=simplifcation)
        estimator.save(save_filepath)
        best_solution = estimator.solution
        print("Program simplified:\n", best_solution.program.code.pretty_str())


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
    parser = argparse.ArgumentParser()
    parser.add_argument('-p', '--pop_size', help="PushGP population size (default 50).", nargs='?', default=1, type=int)
    parser.add_argument('-g', '--gens', help="PushGP generations (default 10).", nargs='?', default=1, type=int)
    parser.add_argument('-s', '--steps', help="Steps for the CA (default 25).", nargs='?', default=25, type=int)
    parser.add_argument('-c', '--cut_size', help="Number of samples for each label (default 10).", nargs='?',
                        default=10, type=int)
    parser.add_argument('-d', '--digits', help="Array of digits (default 1,2).", nargs='?', default='1,2', type=str)
    parser.add_argument('-lf', '--load_file', help="File to write to.", nargs='?', default='test-1.txt', type=str)
    parser.add_argument('-sf', '--save_file', help="File to save to.", nargs='?', default='test-1.txt', type=str)
    parser.add_argument('-m', '--mode', help="Training or testing mode.", nargs='?', default='training', type=str)
    parser.add_argument('-r', '--random', help="Use a random sample of dataset.", nargs='?', default='False', type=str)
    parser.add_argument('-simplify', '--simplification', help="Number of simplification steps", nargs='?', default=0,
                        type=int)
    parser.add_argument('-stacks', '--stacks', help="Which stacks to include.", nargs='?', default="float", type=str)
    parser.add_argument('-pic', '--picture', help="Whether or not to output a picture after testing.", nargs='?',
                        default="false", type=str)

    args = parser.parse_args()
    digits = [int(item) for item in args.digits.split(',')]
    shuffle = args.random.lower() == 'true'
    picture = args.picture.lower() == 'true'
    stacks = set([item.strip() for item in re.split(', |,|; | |;', args.stacks)])

    mnist_pysh_ca(
        mode=args.mode,
        load_filepath=args.load_file,
        save_filepath=args.save_file,
        pop_size=args.pop_size,
        gens=args.gens,
        steps=args.steps,
        cut_size=args.cut_size,
        digits=digits,
        shuffle=shuffle,
        simplifcation=args.simplification,
        stacks=stacks,
        picture=picture
    )
