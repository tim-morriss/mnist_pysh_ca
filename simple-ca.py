import numpy as np
from cellular_automaton import CellularAutomaton, CAWindow
from ca_animate import CAAnimate
from mnist_ca import MNISTCA
from load_datasets import LoadDatasets


class DrawCA(CAWindow):

    def __init__(self, cellular_automaton: CellularAutomaton):
        self._cellular_automaton = cellular_automaton
        self.states = []

    def run(self, evolutions_per_step=1, last_evolution_step=100, **kwargs):
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
            self.states[evolution_step - 1][coordinate[0]][coordinate[1]] = cell.state[0]


if __name__ == '__main__':
    train_x, train_y = LoadDatasets.load_mnist_tf(return_digit=10, plot_digit=10)
    train_x = [np.resize(train_x[i], (28, 28)) for i in range(train_x.shape[0])]  # Reshape X
    x = DrawCA(MNISTCA(train_x, train_y)).run(last_evolution_step=10)
    CAAnimate.animate_ca(x, 'output-1.gif')

