from typing import Union, Callable

import numpy as np
import pandas as pd

from pyshgp.gp.estimators import PushEstimator
from pyshgp.gp.genome import GeneSpawner
from pyshgp.push.instruction_set import InstructionSet
from pyshgp.tap import tap
from pyshgp.utils import list_rindex
from pyshgp.push.program import ProgramSignature, Program
from pyshgp.gp.evaluation import Evaluator, FunctionEvaluator
from pyshgp.push.interpreter import PushInterpreter
from pyshgp.validation import check_X_y

from mnist_ca import MNISTCA, RunCA


class ErrorFunction:

    """

    """

    @staticmethod
    def mnist_error_function(
            program: Program,
            X, y,
            interpreter: PushInterpreter,
            steps: int = 100) -> float:
        """
        Takes a single X and y value and runs them through a CA.

        :param program: a individual push program
        :param X: one array of x
        :param y: its corresponding label
        :param interpreter: the PushInterpreter
        :param steps: how many steps of the CA are needed
        :return: returns an array with the average
        """
        # print(program.pretty_str())
        X = np.expand_dims(X, axis=0)
        output = RunCA(MNISTCA(X, y, program, interpreter)).run(last_evolution_step=steps)
        # print("Output from CA: {0}".format(output.shape))
        average = np.average(output[-1].reshape(-1))
        # print("Average: {0}".format(average))
        return average


class MNISTEstimator(PushEstimator):

    def __init__(self, spawner: GeneSpawner, steps: int, *args, **kwargs):
        super().__init__(spawner, **kwargs)
        self.steps = steps

    @tap
    def fit(self, X, y):
        """Run the search algorithm to synthesize a push program.

                Parameters
                ----------
                X : pandas dataframe of shape = [n_samples, n_features]
                    The training input samples.
                y : list, array-like, or pandas dataframe.
                    The target values (class labels in classification, real numbers in
                    regression). Shape = [n_samples] or [n_samples, n_outputs]

                """
        X, y, arity, y_types = check_X_y(X, y)
        # Set the output types for the pysh programs.
        output_types = [self.interpreter.type_library.push_type_for_type(t).name for t in y_types]
        if self.last_str_from_stdout:
            ndx = list_rindex(output_types, "str")
            if ndx is not None:
                output_types[ndx] = "stdout"
        # Create signature for the estimator
        self.signature = ProgramSignature(arity=1, output_stacks=output_types, push_config=self.push_config)
        # Initialise the evaluator with error function, x and y, interpreter and steps.
        self.evaluator = CustomFunctionEvaluator(
            error_function=ErrorFunction(),
            X=X, y=y,
            interpreter=self.interpreter,
            steps=self.steps
        )
        self._build_search_algo()
        self.solution = self.search.run()
        self.search.config.tear_down()


class CustomFunctionEvaluator(Evaluator):

    def __init__(self,
                 error_function: ErrorFunction,
                 X, y,
                 interpreter: PushInterpreter = "default",
                 penalty: float = 1e6,
                 steps: int = 100):
        super().__init__(interpreter, penalty)
        self.X = pd.DataFrame(X)
        self.y = pd.DataFrame(y)
        self.error_function = error_function
        self.steps = steps

    @tap
    def evaluate(self, program: Program, optimal_number: float = 1) -> np.ndarray:
        """
        Works through each sample in the input and runs it with the individual (program).

        :param program: a push program to be evaluated.
        :param optimal_number: the number to optimise to.
        :return: the error vector
        """
        super().evaluate(program)
        errors = []
        # Work through the input and run error function on each one
        for ndx in range(self.X.shape[0]):
            input_x = self.X.iloc[ndx]
            input_y = self.y.iloc[ndx]
            # print(input_x)
            output = self.error_function.mnist_error_function(
                program,
                input_x, input_y,
                self.interpreter,
                self.steps
            )
            errors.append(optimal_number - output)
            # print(errors)
        return np.array(errors).flatten()

# class FitnessMNIST(Evaluator):
#
#     def __init__(self, X, y, interpreter: PushInterpreter, penalty: float = 1e6):
#         super().__init__(penalty=penalty)
#         self.X = pd.DataFrame(X)
#         self.y = pd.DataFrame(y)
#         self.interpreter = interpreter
#
#     @tap
#     def evaluate(self, program: Program) -> np.ndarray:
#         super().evaluate(program)
#         errors = []
#         for ndx in range(self.X.shape[0]):
#             inputs = self.X.iloc[ndx].to_list()
#             expected = self.y.iloc[ndx].to_list()
#             actual = self.interpreter.run(program, inputs)
#             errors.append(self.default_error_function(actual, expected))
#         return np.array(errors).flatten()
#
#
# class MNISTInterpreter(PushInterpreter):
#
#     def __init__(self,
#                  instruction_set: Union[InstructionSet, str] = "core",
#                  reset_on_run: bool = True):
#         super().__init__(instruction_set, reset_on_run)
#
#
#     @tap
#     def run(self,
#             program: Program,
#             inputs: list,
#             print_tract: bool = False) -> list:
#         pass

