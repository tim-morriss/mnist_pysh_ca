import numpy as np

from typing import List
from numbers import Number
from pysh_ca.ca.pysh_ca import PyshCA
from pysh_ca.ca.run_ca import RunCA
from pyshgp.push.program import Program
from pyshgp.push.interpreter import PushInterpreter
from pysh_ca.pyshgp.class_function import ClassFunction


class CAErrorFunction:
    """
    Custom ErrorFunction for use with CustomFunctionEvaluator.
    """

    def __init__(self, class_function: ClassFunction):
        self.last_ca_grid = None
        self.class_function = class_function

    def ca_error_function(
            self,
            dimensions: List[int],
            program: Program,
            X: np.ndarray,
            y: np.ndarray,
            interpreter: PushInterpreter,
            steps: int = 100) -> Number:
        """
        Takes a single X and y value and runs them through a CA.

        Parameters
        ----------
        dimensions: List[int]
            Dimensions of the CA
        program: Program
            An individual push program
        X: numpy array
            Np.array of one mnist digit
        y: numpy array
            Label of digit
        interpreter: PushInterpreter
            Interpreter used to run the individual push programs
        steps: int, optional
            The number of steps of the CA to execute per evolution
        """
        X = np.expand_dims(X, axis=0)
        output = RunCA(PyshCA(dimensions, X, y, program, interpreter)).run(last_evolution_step=steps)
        # print("Grid output: \n", output.shape)
        self.last_ca_grid = output
        # Average just the last grid state.
        return self.class_function.classify(output)
