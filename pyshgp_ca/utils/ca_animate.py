import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from typing import Tuple


class CAAnimate:

    @staticmethod
    def animate_ca(x, filepath, interval=10):
        """
        Generates a animated gif of the input x.

        :param x: a list of x steps over time
            :type: list of np.array
        :param filepath: filepath of the output file as str
        :param interval: the interval between the frames in ms
        """
        fig = plt.figure(figsize=x[0].shape)

        ax = plt.axes()
        ax.set_axis_off()

        # Generator function to return an image for each step
        def animate(i):
            ax.clear()  # clear the plot
            ax.set_axis_off()  # disable axis
            # print("x[%s]: %s" % (i, x[i]))
            img = ax.imshow(x[i], interpolation='none', cmap='RdPu')
            return [img]

        # call the animator
        anim = animation.FuncAnimation(fig, animate, len(x), interval=interval, blit=True)
        anim.save(filepath, writer='imagemagick')
