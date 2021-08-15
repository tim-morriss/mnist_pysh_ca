import numpy as np
import tensorflow as tf
import random
from typing import Tuple
from matplotlib import pyplot as plt
from sklearn.utils import shuffle as shuff


class LoadDatasets:

    @staticmethod
    def load_mnist_tf(cut_size: int = None, plot_digit: int = None, return_digit: int = None):
        """
        Loads the mnist dataset from the Tensorflow package.

        Parameters
        ----------
        cut_size: int
            How many samples to return overall.
        plot_digit: int
            Plot a digit from the loaded data set to display using matplotlib.
        return_digit: int
            Return digits only of a specific label.

        Returns
        -------
        np.ndarray
            Numpy array of whole MNIST data set
        """
        (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
        x_train = np.concatenate((x_train, x_test), axis=0)
        y_train = np.concatenate((y_train, y_test), axis=0)
        x_train = np.array(x_train / 255.0, ).astype(np.float32)

        if plot_digit is not None:
            LoadDatasets.plot_mnist(x_train[plot_digit])

        if return_digit is not None:
            return np.expand_dims(x_train[return_digit], axis=0), np.expand_dims(y_train[return_digit], axis=0)
        else:
            if cut_size == 0 or cut_size is None:
                return x_train, y_train
            else:
                return x_train[:cut_size], y_train[:cut_size]

    @staticmethod
    def plot_mnist(image: np.array):
        """
        Plots an image from the MNIST dataset with matplotlib.

        Parameters
        ----------
        image: np.array
            Image to plot
        """
        pixels = image.reshape((28, 28))
        plt.imshow(pixels, cmap='gray')
        plt.show()

    @staticmethod
    def exclusive_digit(x: np.ndarray, y: np.ndarray, number_to_return: int) -> Tuple[np.ndarray, np.ndarray]:
        """
        Return a specific number label only from the MNIST dataset (i.e. only 5s).

        Parameters
        ----------
        x: np.ndarray
            The values of the MNIST dataset.
        y: np.ndarray
            The labels of the MNIST dataset.
        number_to_return: int
            The number to be returned.

        Returns
        -------
        Tuple[np.ndarray, np.ndarray]
            Returns a tuple of data and labels for one digit in the MNIST data set
        """
        y_filter = np.where((y == number_to_return))  # Filter only where y is equal to number_to_return
        return np.array(x[y_filter[0]]), np.array(y[y_filter[0]])  # Apply this to the two numpy arrays

    @staticmethod
    def exclusive_digits(
            x: np.ndarray,
            y: np.ndarray,
            numbers_to_return: list,
            cut_size: int = None,
            shuffle: bool = False) \
            -> Tuple[np.ndarray, np.ndarray]:
        """
        Returns a number of samples belonging to specified labels.

        Parameters
        ----------
        x: np.ndarray
            The values of the MNIST dataset.
        y: np.ndarray
            The labels of the MNISt dataset.
        numbers_to_return: list
            The numbers to return.
        cut_size: int
            How many of each label to return
        shuffle: bool
            Whether or not to shuffle samples before returning

        Returns
        -------
        tuple(np.ndarray, np.ndarray)
            Filtered data (x) and labels (y) arrays
        """
        temp_x = np.empty([0, x.shape[1]])
        temp_y = np.empty([0, y.shape[1]])
        for num in numbers_to_return:
            x_1, y_1 = LoadDatasets.exclusive_digit(x, y, num)
            if shuffle:
                x_1, y_1 = shuff(x_1, y_1)
            if cut_size is not None:
                temp_x = np.append(temp_x, x_1[:cut_size], axis=0)
                temp_y = np.append(temp_y, y_1[:cut_size], axis=0)
            else:
                temp_x = np.append(temp_x, x_1, axis=0)
                temp_y = np.append(temp_y, y_1, axis=0)

        return temp_x, temp_y

    @staticmethod
    def random_digit(x, y):
        """
        Returns a random digit from the provided dataset.

        Parameters
        ----------
        x: np.ndarray
            The values of the MNIST dataset.
        y: np.ndarray
            The labels of the MNIST dataset.

        Returns
        -------
        tuple(np.ndarray, np.ndarray)
            One random digit in the form of a np.ndarray.
        """
        return np.expand_dims(random.choice(x), axis=0), np.expand_dims(random.choice(y), axis=0)
