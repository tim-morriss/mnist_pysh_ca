import random
import numpy as np

from pathlib import Path
from time import gmtime, strftime
from typing import List, Set, Sequence

from pysh_ca.pyshgp.class_function import ClassFunction
from pysh_ca.utils.load_datasets import LoadDatasets
from pyshgp.gp.genome import GeneSpawner
from pyshgp.push.instruction_set import InstructionSet
from pyshgp.gp.selection import Lexicase
from pyshgp.tap import Tap, TapManager
from pysh_ca.pyshgp.ca_estimator import CAEstimator
from pysh_ca.utils.ca_animate import CAAnimate


class MNISTPyshCA:

    @staticmethod
    def mnist_pysh_ca(
            mode: str,
            load_filepath: str,
            save_folder: str,
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
        Function to create and run the pyshGP + CA system.

        Parameters
        ----------
        picture : bool
            Whether to output gif of last seen CA.
        mode: Mode
            Choose between training and testing operation modes
        load_filepath: str
            Filepath used for loading existing models for testing and simplification.
        save_folder: str
            Folder to save to.
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
            Number of simplification steps to do (only works in training and simplify modes).
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

        estimator = CAEstimator(
            spawner=spawner,
            class_function=MNISTClassify(),
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

        def filepath(ending: str, extension: str, extra_string: str = None) -> str:

            name_string = extra_string

            if name_string is None:
                name_string = load_filepath.split('.')[0] + '/'
                Path(save_folder + '/' + name_string).mkdir(parents=True, exist_ok=True)

            output_str = '{0}/{1}{2} -- digits {3} -- {4} steps -- cut_size {5}{6}.{7}'.format(
                save_folder, name_string, strftime("%Y-%m-%d %H-%M", gmtime()),
                digits, steps, cut_size, ending, extension
            )

            return output_str

        if mode == 'testing':
            estimator.load(load_filepath)
            score = estimator.score(X=X, y=y)
            print("\n Error vector: \n", score)
            print("Total Error: \n", score.sum())
            output = estimator.evaluator.output
            print("CA output: \n", output)

            if picture:
                CAAnimate.animate_ca(
                    estimator.evaluator.error_function.last_ca_grid,
                    filepath(" -- CA output", "gif")
                )
            with open(filepath(" -- test", "txt"), 'w') as f:
                f.write("Error vector: \n {0}".format(score))
                f.write("\nTotal Error: \n {0}".format(score.sum()))
                f.write("\nCA output: \n {0}".format(output))

        if mode == 'training':
            TapManager.register("pyshgp.gp.search.SearchAlgorithm.step", PopulationTap())
            estimator.fit(X=X, y=y)
            estimator.save(filepath("", "json", ""))
            best_solution = estimator.solution
            print("Program:\n", best_solution.program.code.pretty_str())
            if simplifcation > 0:
                estimator.simplify(X=X, y=y, simplification_steps=simplifcation)
                estimator.save(filepath(" -- simplified", "json", ""))
                print("Program simplified:\n", best_solution.program.code.pretty_str())

        if mode == 'simplify':
            estimator.load(load_filepath)
            estimator.simplify(X=X, y=y, simplification_steps=simplifcation)
            estimator.save(filepath(" -- simplified", "json", ""))
            best_solution = estimator.solution
            print("Program simplified:\n", best_solution.program.code.pretty_str())


class MNISTClassify(ClassFunction):

    def classify(self, ca_output: Sequence):
        average = np.average(ca_output[-1].reshape(-1))
        return average


class PopulationTap(Tap):

    def pre(self, id: str, args, kwargs, obj=None):
        """Print population stats before the next step of the run."""
        search = args[0]
        best_individual = search.population.best()
        print()
        print("Generation:", search.generation)
        print("Best Program:", best_individual.program.pretty_str())
        print("Best Error Vector:", best_individual.error_vector)
        print("Best Total Error:", best_individual.total_error)
