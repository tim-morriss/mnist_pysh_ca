import numpy as np

from enum import Enum
from cellular_automaton import CellularAutomaton, EdgeRule, MooreNeighborhood
from pyshgp.push.interpreter import PushInterpreter
from typing import Sequence, List
from pyshgp.push.program import Program
from pysh_ca.ca.ca_init_function import CAInitFunction


class PyshCA(CellularAutomaton):

    def __init__(self,
                 dimension: List[int],
                 init_function: CAInitFunction,
                 x: np.ndarray,
                 y: np.ndarray,
                 program: Program,
                 interpreter: PushInterpreter,
                 edge_rule: EdgeRule = EdgeRule.FIRST_AND_LAST_CELL_OF_DIMENSION_ARE_NEIGHBORS):
        """
        Extended cellular automata to accommodate PyshGP rules.
        Designed to work with labeled numeric data sets.

        Parameters
        ----------
        dimension: list[int]
            dimension of the CA
        x: np.array
            Data input
        y: np.array
            Label input
        program: Program
            Pyshgp program used to update cells.
        interpreter: PushInterpreter
            PushInterpreter used for running programs.
        edge_rule: EdgeRule, optional
            The rule of the CA when dealing with edge cells.
            Options are:
            IGNORE_EDGE_CELLS,
            IGNORE_MISSING_NEIGHBORS_OF_EDGE_CELLS, and
            FIRST_AND_LAST_CELL_OF_DIMENSION_ARE_NEIGHBORS.
        """

        self.x = x
        self.x = self.x.reshape([-1] + dimension)   # Reshape to grid shape
        self.y = y
        self.program = program
        self.interpreter = interpreter
        self.init_function = init_function
        super().__init__(
            dimension=dimension,
            neighborhood=MooreNeighborhood(edge_rule)
        )

    def init_cell_state(self, cell_coordinate: Sequence) -> Sequence:
        """
        Initialises the cells in the CA grid.

        Parameters
        ----------
        cell_coordinate: Sequence
            Coordinate of the cell to initialise.

        Returns
        ----------
        sequence
            The initialised cell.
        """
        init = self.init_function.ca_init_function(self.x, self.y, cell_coordinate)
        return [init]

    def evolve_rule(self, last_cell_state: Sequence, neighbors_last_states: Sequence) -> Sequence:
        """
        The evolution rule for the CA grid. This runs for each cell in the gird once per evolution step.

        Parameters
        ----------
        last_cell_state: Sequence
            The state of the cell at the preceding step of the CA.
        neighbors_last_states: Sequence
            The states of the neighbours around it (defined by the neighbourhood setting).

        Returns
        ----------
        Sequence
            The new state of the cell.
        """

        # Use the current cell state as the input for the push program which outputs the next cell state
        neighborhood = last_cell_state + np.array(neighbors_last_states).flatten()
        value = self.interpreter.run(self.program, neighborhood, print_trace=False)  # print_trace for debugging
        if isinstance(value, list):
            value = value[0]
        # Catches Token values from the PushInterpreter (used when output stack is empty), replaces with 0.
        if isinstance(value, Enum):
            value = 0
        return [value]

