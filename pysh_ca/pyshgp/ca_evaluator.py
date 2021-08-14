from typing import List

import numpy as np
import pandas as pd
from pyshgp.gp.evaluation import Evaluator
from pyshgp.push.interpreter import PushInterpreter
from pyshgp.push.program import Program
from pyshgp.tap import tap
from pysh_ca.pyshgp.ca_error_function import CAErrorFunction


class CAEvaluator(Evaluator):

    def __init__(self,
                 dimensions: List[int],
                 error_function: CAErrorFunction,
                 X, y,
                 interpreter: PushInterpreter = "default",
                 penalty: float = 1e6,
                 steps: int = 100):
        """

        Parameters
        ----------
        dimensions : List[int]
            dimensions for the CA
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
        self.output = []
        self.dimensions = dimensions

    @tap
    def evaluate(self, program: Program) -> np.ndarray:
        """
        Works through each sample in the input and runs it with the individual (program).

        Parameters
        ----------
        program: Program
            A push program to be evaluated.

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
            output = self.error_function.ca_error_function(
                self.dimensions,
                program,
                input_x, input_y,
                self.interpreter,
                self.steps
            )
            self.output.append(output)
            errors.append(abs(input_y.to_list()[0] - output))
        return np.array(errors).flatten()
