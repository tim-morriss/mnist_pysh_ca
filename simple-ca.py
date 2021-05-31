# import json
import numpy as np
from typing import Sequence
from cellular_automaton import CellularAutomaton, MooreNeighborhood, CAWindow, EdgeRule
from load_datasets import LoadDatasets
# from pyshgp.push.stack import PushStack
# from pyshgp.push.types import PushIntType
# from pyshgp.push.config import PushConfig


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
        init = (1 if cell > 0 else 0)
        return [init]

    def evolve_rule(self, last_cell_state: Sequence, neighbors_last_states: Sequence) -> Sequence:
        return last_cell_state

    def mnist_load(self):
        train_X, train_y = LoadDatasets.load_mnist(
            'Data/mnist/train-images.idx3-ubyte',
            'Data/mnist/train-labels.idx1-ubyte',
            self.n_size
        )

        train_X = [np.resize(train_X[i], (28, 28)) for i in range(train_X.shape[0])]  # Reshape X

        return train_X, train_y


# def ca_grid(n_size) -> Tuple[np.ndarray, np.ndarray]:
#     train_X, train_y = LoadDatasets.load_mnist(
#         'Data/mnist/train-images.idx3-ubyte',
#         'Data/mnist/train-images.idx3-ubyte',
#         n_size
#     )
#
#     # Reshape X
#     train_X = [np.resize(train_X[i], (28, 28)) for i in range(train_X.shape[0])]
#
#     return np.array(train_X), np.array(train_y)
#     # print("train_x: %s \n train_y: %s" % (train_X, train_y))


if __name__ == '__main__':
    CAWindow(
        cellular_automaton=MNISTCA(1),
        window_size=(1000, 830)
    ).run(evolutions_per_second=1)
    # x, y = ca_grid(2)
    # print(type(x[0]))
    #
    # print("(0,0): %s" % x[0][0][0])  # Finding (0,0)
    # print("(0,1): %s" % x[0][0][1])  # Finding (0,1)

    # push_stack = PushStack(PushIntType(), PushConfig())
    # push_stack.append(4)
    # push_stack.append(5)
    # print(push_stack)