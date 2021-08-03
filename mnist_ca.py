from enum import Enum

import numpy as np
from cellular_automaton import CellularAutomaton, CAWindow, EdgeRule, MooreNeighborhood
from pyshgp.push.interpreter import PushInterpreter

from typing import Sequence

from pyshgp.utils import Token

from load_datasets import LoadDatasets
from pyshgp.push.program import Program


class MNISTCA(CellularAutomaton):

    def __init__(self,
                 x, y,
                 program: Program,
                 interpreter: PushInterpreter,
                 edge_rule=EdgeRule.FIRST_AND_LAST_CELL_OF_DIMENSION_ARE_NEIGHBORS):
        """
        Modified CellularAutomaton class that specifies cell states and evolve rules for the MNIST dataset.
        Takes one MNIST digit and one Pyshgp program at a time.

        Parameters
        ----------
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
        self.x = self.x.reshape([-1, 28, 28])   # Reshape to grid shape
        # print("self x: {0}".format(self.x.shape))
        self.y = y
        self.program = program
        self.interpreter = interpreter
        super().__init__(
            dimension=[27, 27],  # 28 x 28 pixels
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

        cell = self.x[0][cell_coordinate[0]][cell_coordinate[1]]
        init = cell
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

        # print(self.x.shape)
        # temp_x = self.x.reshape(784).tolist()
        # temp_x = last_cell_state
        # print("last_cell_state: {0} and of type: {1}".format(last_cell_state, type(last_cell_state)))

        # Use the current cell state as the input for the push program which outputs the next cell state
        neighborhood = last_cell_state + np.array(neighbors_last_states).flatten()
        value = self.interpreter.run(self.program, neighborhood, print_trace=False)  # print_trace for debugging
        # print("Value: {0}".format(value))
        if isinstance(value, list):
            value = value[0]
        # Catches Token values from the PushInterpreter (used when ouput stack is empty), replaces with 0.
        if isinstance(value, Enum):
            value = 0
            # print("code gets here and value is: {0}".format(value))
        return [value]


class RunCA(CAWindow):
    """
    Class for running the CA. Adapted from CAWindow but avoids using PyGame for visualisation.

    Saves each evolution step in self.states.
    """

    def __init__(self, cellular_automaton: CellularAutomaton):
        """
        Designed to use run a MNISTCA object, not currently applicable for any other CellularAutomaton objects.

        Parameters
        ----------
        cellular_automaton: CellularAutomaton
            CellularAutomaton object to be run.
        """

        self._cellular_automaton = cellular_automaton
        self.states = np.empty(0)

    def run(self, evolutions_per_step: int = 1, last_evolution_step: int = 100, **kwargs) -> Sequence:
        """
        Runs the CellularAutomaton for a number of steps.

        Parameters
        ----------
        evolutions_per_step: int
            How many evolutions should each cell undertake per step.
        last_evolution_step: int
            How many steps to execute.
        kwargs:

        Returns
        ----------
        Sequence
            Sequence of cell states.
        """

        # Initialise self.states with the specified number of empty arrays.
        self.states = np.zeros([last_evolution_step, 28, 28], dtype=np.float32)
        while self._not_at_the_end(last_evolution_step):
            self._cellular_automaton.evolve(evolutions_per_step)
            self.update_states(self._cellular_automaton.evolution_step)
        return self.states

    def get(self):
        return self._cellular_automaton.get_cells()

    def update_states(self, evolution_step: int):
        """
        Updates the states after evolution.

        Parameters
        ----------
        evolution_step: int
            Current evolution step.
        """

        # For each cell and co-ordinate in the CA update the corresponding point in self.states.
        for coordinate, cell in self._cellular_automaton.cells.items():
            # print("Coordinate: %s, state: %s" % (coordinate, cell.state))
            # if self.states[evolution_step - 1] is None:
                # self.states[evolution_step - 1] = np.zeros((28, 28), dtype=np.float32)
            # print("cell state: {0}".format(cell.state[0]))
            self.states[evolution_step - 1][coordinate[0]][coordinate[1]] = cell.state[0]
