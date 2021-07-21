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
                 update_rule: Program,
                 interpreter: PushInterpreter,
                 edge_rule=EdgeRule.FIRST_AND_LAST_CELL_OF_DIMENSION_ARE_NEIGHBORS):
        """
        Modified CellularAutomaton class that specifies cell states and evolve rules for the MNIST dataset.
        :param edge_rule: The rule of the CA when dealing with edge cells.
            Options are:
            IGNORE_EDGE_CELLS,
            IGNORE_MISSING_NEIGHBORS_OF_EDGE_CELLS, and
            FIRST_AND_LAST_CELL_OF_DIMENSION_ARE_NEIGHBORS.
        """
        self.x = x
        self.x = self.x.reshape([-1, 28, 28])
        print("self x: {0}".format(self.x.shape))
        self.y = y
        self.update_rule = update_rule
        self.interpreter = interpreter
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
        :return: The new state of the cell (sequence).
        """
        value = self.interpreter.run(self.update_rule, self.x)
        if isinstance(value, Enum):
            value = 0
            print("code gets here and value is: {0}".format(value))
        return [value]


class DrawCA(CAWindow):

    def __init__(self, cellular_automaton: CellularAutomaton):
        self._cellular_automaton = cellular_automaton
        self.states = []

    def run(self, evolutions_per_step=1, last_evolution_step=100, **kwargs):
        if not self.states:
            self.states = [None] * last_evolution_step
        while self._not_at_the_end(last_evolution_step):
            self._cellular_automaton.evolve(evolutions_per_step)
            self.update_states(self._cellular_automaton.evolution_step)
        return self.states

    def get(self):
        return self._cellular_automaton.get_cells()

    def update_states(self, evolution_step):
        for coordinate, cell in self._cellular_automaton.cells.items():
            # print("Coordinate: %s, state: %s" % (coordinate, cell.state))
            if self.states[evolution_step - 1] is None:
                self.states[evolution_step - 1] = np.zeros((28, 28), dtype=np.float32)
            print("cell state: {0}".format(cell.state[0]))
            self.states[evolution_step - 1][coordinate[0]][coordinate[1]] = cell.state[0]


def test_number(number, cut_size, evolution_steps):
    print()
    print("--------------------------------------------")
    print("Loading MNIST dataset...")
    train_x, train_y = LoadDatasets.load_mnist_tf(cut_size=cut_size)
    print("Filter MNIST images to only include the label %s" % number)
    x_filter, y_filter = LoadDatasets.exclusive_digit(train_x, train_y, number_to_return=number)
    average_intensities = np.zeros(0)
    i = 0
    for (x, y) in zip(x_filter, y_filter):
        # print()
        # print("Evolving the %s label..." % i)
        x = np.expand_dims(x, axis=0)
        y = np.expand_dims(y, axis=0)
        evolution = DrawCA(MNISTCA(x, y)).run(last_evolution_step=evolution_steps)
        average = np.average(evolution[-1].reshape(-1))
        print("Average for %s was %s" % (i, average))
        average_intensities = np.append(average_intensities, average)
        i += 1
    return average_intensities


def average_one_number(num, cut, steps):
    filename = "average_%s.txt" % num
    open(filename, 'w').close()  # Clear text file
    averages = test_number(num, cut, steps)
    with open(filename, "a") as averages_output:
        with np.printoptions(threshold=np.inf):
            print("Writing to file...")
            averages_output.write("\n -------------------------------------------- \n")
            averages_output.write("Averages for %s is: \n %s" % (num, averages))
            averages_output.write("\n Total average is %s" % np.average(averages))
            print("Total average is %s" % np.average(averages))
            averages_output.write("\n -------------------------------------------- \n")


if __name__ == '__main__':
    for i in range(0, 10):
        average_one_number(i, None, 250)

    # open('averages.txt', 'w').close()  # Erase textfile
    # for i in range(0, 10):
    #     averages = test_number(i, None, 250)
    #     with open("averages.txt", "a") as averages_output:
    #         with np.printoptions(threshold=np.inf):
    #             averages_output.write("\n -------------------------------------------- \n")
    #             averages_output.write("Averages for %s is: \n %s" % (i, averages))
    #             averages_output.write("\n Total average is %s" % np.average(averages))
    #             averages_output.write("\n -------------------------------------------- \n")

    # train_x, train_y = LoadDatasets.load_mnist_tf()
    # x_filter, y_filter = LoadDatasets.exclusive_digit(train_x, train_y, number_to_return=5)
    # train_x, train_y = LoadDatasets.random_digit(x_filter, y_filter)
    # # print(train_x, train_y)
    # # train_x, train_y = LoadDatasets.load_mnist_tf(return_digit=10, plot_digit=10)
    # train_x = [np.resize(train_x[i], (28, 28)) for i in range(train_x.shape[0])]  # Reshape X
    # x = DrawCA(MNISTCA(train_x, train_y)).run(last_evolution_step=250)
    # CAAnimate.animate_ca(x, 'output-1.gif')
