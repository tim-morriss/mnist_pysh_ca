from collections import Sequence

import numpy as np
from cellular_automaton import CAWindow, CellularAutomaton


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
            self.states[evolution_step - 1][coordinate[0]][coordinate[1]] = cell.state[0]
