import numpy as np
import pandas as pd

from pyshgp.gp.estimators import PushEstimator
from pyshgp.gp.genome import GeneSpawner
from pyshgp.tap import tap
from pyshgp.utils import list_rindex
from pyshgp.push.program import ProgramSignature, Program
from pyshgp.gp.evaluation import Evaluator
from pyshgp.push.interpreter import PushInterpreter
from pyshgp.validation import check_X_y

from mnist_ca import MNISTCA, RunCA


class ErrorFunction:

    """
    Custom ErrorFunction for use with CustomFunctionEvaluator.
    """

    @staticmethod
    def mnist_error_function(
            program: Program,
            X, y,
            interpreter: PushInterpreter,
            steps: int = 100) -> float:
        """
        Takes a single X and y value and runs them through a CA.

        Parameters
        ----------
        program: Program
            An individual push program
        X: numpy array
            Np.array of one mnist digit
        y: numpy array
            Label of digit
        interpreter: PushInterpreter
            Interpreter used to run the individual push programs
        steps int, optional
            The number of steps of the CA to execute per evolution
        """
        # print(program.pretty_str())
        X = np.expand_dims(X, axis=0)
        output = RunCA(MNISTCA(X, y, program, interpreter)).run(last_evolution_step=steps)
        # print("Output from CA: {0}".format(output.shape))
        # Average just the last grid state.
        average = np.average(output[-1].reshape(-1))
        # print("Average: {0}".format(average))
        return average


class MNISTEstimator(PushEstimator):

    def __init__(self, spawner: GeneSpawner, steps: int, *args, **kwargs):
        """

        Parameters
        ----------
        spawner: GeneSpawner
            Used to spawn individuals.
        steps: int
            Number of steps of evolution of the CA grid.
        args
        kwargs
        """
        super().__init__(spawner, **kwargs)
        self.steps = steps

    @tap
    def fit(self, X, y):
        """
        Runs and optimises the population of Push programs.
        Optimises based on error function.

        Parameters
        ----------
        X: pandas dataframe of shape = [n_samples, n_features]
            The training input samples.
        y: list, array-like, or pandas dataframe.
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

    @tap
    def score(self, X, y):
        """
        Allows scoring of loaded individual

        Parameters
        ----------
        X: pandas dataframe of shape = [n_samples, n_features]
            The training input samples.
        y: list, array-like, or pandas dataframe.
            The target values (class labels in classification, real numbers in
            regression). Shape = [n_samples] or [n_samples, n_outputs]

        Returns
        -------
        np.array
            Error vector

        """
        X, y, arity, y_types = check_X_y(X, y)
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
        print("Individual being tested: \n", self.solution.program.code.pretty_str())
        errors = self.evaluator.evaluate(self.solution.program)
        return np.array(errors)


class CustomFunctionEvaluator(Evaluator):

    def __init__(self,
                 error_function: ErrorFunction,
                 X, y,
                 interpreter: PushInterpreter = "default",
                 penalty: float = 1e6,
                 steps: int = 100):
        """

        Parameters
        ----------
        error_function: ErrorFunction object
        X: pandas dataframe of shape = [n_samples, n_features]
            The training input samples.
        y: list, array-like, or pandas dataframe.
            The target values (class labels in classification, real numbers in
            regression). Shape = [n_samples] or [n_samples, n_outputs]
        interpreter: PushInterpreter, optional
            PushInterpreter used to run program and get their output. Default is
            an interpreter with the default configuration and all core instructions
            registered.
        penalty: float, optional
            When a program's output cannot be evaluated on a particular case, the
            penalty error is assigned. Default is 5e5.
        steps: int, optional
            Number of steps of evolution of the CA grid.
        """
        super().__init__(interpreter, penalty)
        self.X = pd.DataFrame(X)
        self.y = pd.DataFrame(y)
        self.error_function = error_function
        self.steps = steps

    @tap
    def evaluate(self, program: Program, optimal_number: float = None) -> np.ndarray:
        """
        Works through each sample in the input and runs it with the individual (program).

        Parameters
        ----------
        program: Program
            A push program to be evaluated.
        optimal_number: float, optional
            The number to optimise to.

        Returns
        -------
        np.array
            The error vector for every sample in the dataset.
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

            # Idea is to try and optimise the average of the grid after the evolutions to the y label.
            # This might be redundant as not sure if optimal_number strategy is useful
            # print("Error: {0}".format(abs(input_y.to_list()[0] - output)))
            # print("Input_y: {0}".format(input_y.to_list()[0]))
            if not optimal_number:
                errors.append(abs(input_y.to_list()[0] - output))
            else:
                errors.append(optimal_number - output)
        # print(errors)
        return np.array(errors).flatten()

