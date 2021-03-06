from typing import List

import numpy as np

from pyshgp.gp.estimators import PushEstimator
from pyshgp.gp.genome import GeneSpawner, GenomeSimplifier
from pyshgp.gp.individual import Individual
from pyshgp.tap import tap
from pyshgp.utils import list_rindex
from pyshgp.push.program import ProgramSignature
from pyshgp.validation import check_X_y
from pysh_ca.ca.ca_init_function import CAInitFunction
from pysh_ca.pyshgp.ca_error_function import CAErrorFunction
from pysh_ca.pyshgp.class_function import ClassFunction
from pysh_ca.pyshgp.ca_evaluator import CAEvaluator


class CAEstimator(PushEstimator):

    def __init__(
            self,
            dimensions: List[int],
            spawner: GeneSpawner,
            class_function: ClassFunction,
            init_function: CAInitFunction,
            steps: int,
            *args, **kwargs):
        """

        Parameters
        ----------
        spawner: GeneSpawner
            Used to spawn individuals.
        class_function : ClassFunction
            Function used to classify CA output.
        steps: int
            Number of steps of evolution of the CA grid.
        args
        kwargs
        """
        super().__init__(spawner, *args, **kwargs)
        self.steps = steps
        self.class_function = class_function
        self.init_function = init_function
        self.dimensions = dimensions

    def _initialise_signature(self, X, y):
        X, y, arity, y_types = check_X_y(X, y)
        output_types = [self.interpreter.type_library.push_type_for_type(t).name for t in y_types]
        if self.last_str_from_stdout:
            ndx = list_rindex(output_types, "str")
            if ndx is not None:
                output_types[ndx] = "stdout"
        # Create signature for the estimator
        self.signature = ProgramSignature(arity=1, output_stacks=output_types, push_config=self.push_config)

    def _initialise_evaluator(self, X, y):
        # Initialise the evaluator with error function, x and y, interpreter and steps.
        self.evaluator = CAEvaluator(
            self.dimensions,
            error_function=CAErrorFunction(self.class_function, self.init_function),
            X=X, y=y,
            interpreter=self.interpreter,
            steps=self.steps
        )

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
        self._initialise_signature(X, y)
        self._initialise_evaluator(X, y)
        self._build_search_algo()
        self.solution = self.search.run()
        self.search.config.tear_down()

    @tap
    def score(self, X, y):
        """
        Scores loaded program.

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
        if self.signature is None:
            self._initialise_signature(X, y)

        if self.evaluator is None:
            self._initialise_evaluator(X, y)

        print("\n Individual being tested: \n", self.solution.program.code.pretty_str())
        errors = self.evaluator.evaluate(self.solution.program)
        return np.array(errors)

    @tap
    def simplify(self, X, y, simplification_steps: int = 2000):
        """
        Simplifies a loaded program.

        Parameters
        ----------
       X: pandas dataframe of shape = [n_samples, n_features]
            The training input samples.
        y: list, array-like, or pandas dataframe.
            The target values (class labels in classification, real numbers in
            regression). Shape = [n_samples] or [n_samples, n_outputs]
        simplification_steps: int
            Number of simplification steps to conduct
        """
        if self.signature is None:
            self._initialise_signature(X, y)

        if self.evaluator is None:
            self._initialise_evaluator(X, y)

        simplifier = GenomeSimplifier(
            self.evaluator,
            self.signature
        )
        simp_genome, simp_error_vector = simplifier.simplify(
            self.solution.genome,
            self.solution.error_vector,
            simplification_steps
        )
        simplified_best = Individual(simp_genome, self.signature)
        simplified_best.error_vector = simp_error_vector
        self.solution = simplified_best
