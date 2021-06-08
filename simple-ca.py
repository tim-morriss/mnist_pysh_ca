# import json
import itertools

import numpy as np
# import pygame
from typing import Sequence
from cellular_automaton import CellularAutomaton, MooreNeighborhood, CAWindow, EdgeRule
# from cellular_automaton.display import PygameEngine
from load_datasets import LoadDatasets
from ca_animate import CAAnimate


class MNISTCA(CellularAutomaton):

    def __init__(self, n_size):
        self.n_size = n_size
        self.x, self.y = self.mnist_load()
        super().__init__(
            dimension=[27, 27],  # 28 x 28 pixels
            neighborhood=MooreNeighborhood(EdgeRule.FIRST_AND_LAST_CELL_OF_DIMENSION_ARE_NEIGHBORS)
        )

    def init_cell_state(self, cell_coordinate: Sequence) -> Sequence:
        cell = self.x[0][cell_coordinate[0]][cell_coordinate[1]]
        init = cell
        return [init]

    def evolve_rule(self, last_cell_state: Sequence, neighbors_last_states: Sequence) -> Sequence:
        alive_neighbours = []
        for n in neighbors_last_states:
            if n[0] > 0:
                alive_neighbours.append(1)
        if len(alive_neighbours) > 3:
            return [1]
        else:
            return [0]

    def mnist_load(self):
        train_X, train_y = LoadDatasets.load_mnist_tf(cut_size=self.n_size, plot_digit=None)

        train_X = [np.resize(train_X[i], (28, 28)) for i in range(train_X.shape[0])]  # Reshape X

        # print(train_X[0])
        # print(train_X[0].shape)

        return train_X, train_y


class DrawCA(CAWindow):

    def __init__(self, cellular_automaton: CellularAutomaton):
        self._cellular_automaton = cellular_automaton
        self.states = []

    def run(self, evolutions_per_step=1, last_evolution_step=100, **kwargs):
        self.states = [None] * last_evolution_step
        while self._not_at_the_end(last_evolution_step):
            self._cellular_automaton.evolve(evolutions_per_step)
            # self.print_process_info(self._cellular_automaton.evolution_step)
            self.update_states(self._cellular_automaton.evolution_step)
        return self.states

    def get(self):
        return self._cellular_automaton.get_cells()

    # def print_process_info(self, evolution_step):
    #     print("Evolution step: %s" % evolution_step)

    def update_states(self, evolution_step):
        for coordinate, cell in self._cellular_automaton.cells.items():
            # print("Coordinate: %s, state: %s" % (coordinate, cell.state))
            if self.states[evolution_step - 1] is None:
                self.states[evolution_step - 1] = np.zeros((28, 28), dtype=np.float32)
            self.states[evolution_step - 1][coordinate[0]][coordinate[1]] = cell.state[0]


if __name__ == '__main__':
    x = DrawCA(MNISTCA(1)).run(last_evolution_step=100)
    # print(x)
    CAAnimate.animate_ca(x, 'output-1.gif')
    # train_x, train_y = LoadDatasets.load_mnist_tf(10)
    # print(train_x)
    # CAAnimate.animate_ca(train_x, 'output.gif')
