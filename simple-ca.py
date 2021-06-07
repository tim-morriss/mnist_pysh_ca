# import json
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
            dimension=[27, 27],  # 28 x 28 pixels with 255 stages of intensity
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
        if len(alive_neighbours) > 2:
            return [1]
        else:
            return [0]

    def mnist_load(self):
        train_X, train_y = LoadDatasets.load_mnist_tf(cut_size=self.n_size, plot_digit=None)

        train_X = [np.resize(train_X[i], (28, 28)) for i in range(train_X.shape[0])] # Reshape X

        print(train_X[0].shape)

        return train_X, train_y


if __name__ == '__main__':
    train_x, train_y = LoadDatasets.load_mnist_tf(100)
    CAAnimate.animate_ca(train_x, 'output.gif')