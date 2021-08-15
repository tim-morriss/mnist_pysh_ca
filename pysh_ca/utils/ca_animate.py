import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np


class CAAnimate:

    @staticmethod
    def animate_ca(x: np.ndarray, filepath: str, interval: int = 10):
        """
        Generates a animated gif of the input x.

        Parameters
        ----------
        x: np.ndarray
            A list of x steps over time
        filepath: str
            Filepath of the output file as string
        interval: int
            The interval between frame in ms
        """
        fig = plt.figure(figsize=x[0].shape)

        ax = plt.axes()
        ax.set_axis_off()

        # Generator function to return an image for each step
        def animate(i):
            ax.clear()  # clear the plot
            ax.set_axis_off()  # disable axis
            # print("x[%s]: %s" % (i, x[i]))
            img = ax.imshow(x[i], interpolation='none', cmap='binary')
            return [img]

        # call the animator
        anim = animation.FuncAnimation(fig, animate, len(x), interval=interval, blit=True)
        anim.save(filepath, writer='imagemagick')
