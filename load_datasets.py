import numpy as np
import tensorflow as tf
from typing import Tuple
from struct import unpack
from matplotlib import pyplot as plt


class LoadDatasets:

    @staticmethod
    def load_mnist(image_file: str, label_file: str, cut_size: int = 0, plot_digit: int = None) -> Tuple[np.ndarray, np.ndarray]:
        """
        Takes MNIST input files (x and y) and parses them into two numpy arrays.
        x is the images and y is the label.

        :param image_file:  Image file path (string).
        :param label_file:  Label file path (string).
        :param cut_size:    How much of the dataset to return.
        :param plot_digit:  The image to plot (int).
        :return:    Tuple of numpy arrays.
        """
        # Open image and label files in binary mode
        images = open(image_file, 'rb')
        labels = open(label_file, 'rb')

        # Get metadata for images
        images.read(4)  # Skip the magic_number
        number_of_images = images.read(4)
        number_of_images = unpack('>I', number_of_images)[0]
        # print("number of images: %s" % number_of_images)
        rows = images.read(4)
        rows = unpack('>I', rows)[0]
        # print("rows %s" % rows)
        cols = images.read(4)
        cols = unpack('>I', cols)[0]
        # print("cols: %s" % cols)

        # Get metadata for labels
        labels.read(4)
        N = labels.read(4)
        N = unpack('>I', N)[0]

        # Get data
        X = np.zeros((N, rows * cols), dtype=np.int64)  # Initialise X
        y = np.zeros(N, dtype=np.int64)  # Initialise y
        for i in range(N):
            for j in range(rows * cols):
                tmp_pixel = images.read(1)
                tmp_pixel = unpack('>B', tmp_pixel)[0]
                X[i][j] = tmp_pixel
            tmp_label = labels.read(1)
            y[i] = unpack('>B', tmp_label)[0]

        images.close()
        labels.close()

        X = np.array(X / 255.0, ).astype(np.float32)

        # Allows a digit to be plotted with matplotlib
        if plot_digit is not None:
            LoadDatasets.plot_mnist(X[plot_digit])

        print("MNIST Loaded with a cut size of: %i" % cut_size)
        if cut_size == 0:
            return X, y
        else:
            return X[:cut_size], y[:cut_size]


    @staticmethod
    def load_mnist_tf(cut_size: int = 0, plot_digit: int = None):
        (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
        x_train = np.array(x_train / 255.0, ).astype(np.float32)
        # x_test = np.array(x_test / 255.0, ).astype(np.float32)

        if plot_digit is not None:
            LoadDatasets.plot_mnist(x_train[plot_digit])

        if cut_size == 0:
            print("Full MNIST loaded")
            return x_train, y_train
        else:
            print("MNIST loaded with a cut size of: %i" % cut_size)
            return x_train[:cut_size], y_train[:cut_size]


    @staticmethod
    def plot_mnist(image: np.array):
        pixels = image.reshape((28, 28))
        plt.imshow(pixels, cmap='gray')
        plt.show()
