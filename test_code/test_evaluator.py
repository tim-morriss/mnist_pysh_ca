import numpy as np
import pandas as pd
from pyshgp.gp.evaluation import Evaluator
from pyshgp.push.interpreter import PushInterpreter, Program
from pyshgp.tap import tap


class TestEvaluator(Evaluator):

    def __init__(self,
                 X, y,
                 interpreter: PushInterpreter = "default",
                 penalty: float = 1e6):
        """Create Evaluator based on a labeled dataset. Inspired by sklearn.

        Parameters
        ----------
        X : list, array-like, or pandas dataframe of shape = [n_samples, n_features]
            The inputs to evaluate each program on.

        y : list, array-like, or pandas dataframe.
            The target values. Shape = [n_samples] or [n_samples, n_outputs]

        interpreter : PushInterpreter or {"default"}
            The interpreter used to run the push programs.

        penalty : float
            If no response is given by the program on a given input, assign this
            error as the error.

        """
        super().__init__(interpreter, penalty)
        self.X = pd.DataFrame(X)
        self.y = pd.DataFrame(y)

    @tap
    def evaluate(self, program: Program) -> np.array:
        """Evaluate the program and return the error vector.

        Parameters
        ----------
        program
            Program (CodeBlock of Push code) to evaluate.

        Returns
        -------
        np.ndarray
            The error vector of the program.

        """
        super().evaluate(program)
        errors = []
        for ndx in range(self.X.shape[0]):
            inputs = self.X.iloc[ndx].to_list()
            expected = self.y.iloc[ndx].to_list()
            actual = self.interpreter.run(program, inputs)
            errors.append(self.default_error_function(actual, expected))
        return np.array(errors).flatten()