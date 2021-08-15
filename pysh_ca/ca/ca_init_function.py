import numpy as np

from abc import ABC, abstractmethod
from numbers import Number
from typing import Sequence


class CAInitFunction(ABC):

    @abstractmethod
    def ca_init_function(self, X: np.ndarray, y: np.ndarray, cell_coordinate: Sequence) -> Number:
        """
        Custom initialisation function of CA using data set.
        Input is one sample from data set, with the output being the cell state.

        Parameters
        ----------
        X: np.array
            Data input
        y: np.array
            Label input
        cell_coordinate: Sequence
            Coordinate of the cell to initialise.

        Returns
        -------
        Number
            Initialised cell state
        """
        raise NotImplementedError
