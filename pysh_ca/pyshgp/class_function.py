from abc import ABC, abstractmethod
from typing import Sequence
from numbers import Number


class ClassFunction(ABC):

    @abstractmethod
    def classify(self, ca_output: Sequence):
        """
        Function should take the output vector of the ca grid
        and create a numeric classification based on this.

        Parameters
        ----------
        ca_output: Sequence
            A vector of all cell states at each time step for the CA grid.

        Returns
        -------
        Number
            The classification for the CA output.
        
        """
        raise NotImplementedError
