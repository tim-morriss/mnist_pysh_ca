import numpy as np
from typing import Sequence
from load_datasets import LoadDatasets
from cellular_automaton import CellularAutomaton, MooreNeighborhood, EdgeRule


class MNISTCA(CellularAutomaton):

    def __init__(self, x, y, edge_rule=EdgeRule.FIRST_AND_LAST_CELL_OF_DIMENSION_ARE_NEIGHBORS):
        """
        Modified CellularAutomaton class that specifies cell states and evolve rules for the MNIST dataset.
        :param edge_rule: The rule of the CA when dealing with edge cells.
            Options are:
            IGNORE_EDGE_CELLS,
            IGNORE_MISSING_NEIGHBORS_OF_EDGE_CELLS, and
            FIRST_AND_LAST_CELL_OF_DIMENSION_ARE_NEIGHBORS.
        """
        self.x, self.y = x, y
        super().__init__(
            dimension=[27, 27],  # 28 x 28 pixels
            neighborhood=MooreNeighborhood(edge_rule)
        )

    def init_cell_state(self, cell_coordinate: Sequence) -> Sequence:
        """
        Initialises the cells.
        :param cell_coordinate: Coordinate of the cell to initialise.
        :return: The initialised cell.
        """
        cell = self.x[0][cell_coordinate[0]][cell_coordinate[1]]
        init = cell
        return [init]

    def evolve_rule(self, last_cell_state: Sequence, neighbors_last_states: Sequence) -> Sequence:
        """
        The evolution rules for the CA.
        :param last_cell_state: The state of the cell at the preceding step of the CA.
        :param neighbors_last_states: The states of the neighbours around it (defined by the neighbourhood setting).
        :return: The new state of the cell.
        """
        alive_neighbours = []
        for n in neighbors_last_states:
            if n[0] > 0:
                alive_neighbours.append(1)
        if len(alive_neighbours) > 3:
            return [1]
        else:
            return [0]
