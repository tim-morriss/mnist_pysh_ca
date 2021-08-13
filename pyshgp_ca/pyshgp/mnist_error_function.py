import numpy as np

from pyshgp_ca.ca.mnist_ca import MNISTCA
from pyshgp_ca.ca.run_ca import RunCA
from pyshgp.push.program import Program
from pyshgp.push.interpreter import PushInterpreter


class MNISTErrorFunction:
    """
    Custom ErrorFunction for use with CustomFunctionEvaluator.
    """

    def __init__(self):
        self.last_ca_grid = None

    def mnist_error_function(
            self,
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
        steps: int, optional
            The number of steps of the CA to execute per evolution
        """
        X = np.expand_dims(X, axis=0)
        output = RunCA(MNISTCA(X, y, program, interpreter)).run(last_evolution_step=steps)
        # print("Grid output: \n", output.shape)
        self.last_ca_grid = output
        # Average just the last grid state.
        average = np.average(output[-1].reshape(-1))
        return average
