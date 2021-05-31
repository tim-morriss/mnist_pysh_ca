import numpy as np
from typing import Tuple
from struct import unpack


class LoadDatasets:

    @staticmethod
    def load_mnist(image_file, label_file, cut_size=0) -> Tuple[np.ndarray, np.ndarray]:
        # Open image and label files in binary mode
        images = open(image_file, 'rb')
        labels = open(label_file, 'rb')

        # Get metadata for images
        images.read(4)  # Skip the magic_number
        number_of_images = images.read(4)
        number_of_images = unpack('>I', number_of_images)[0]
        rows = images.read(4)
        rows = unpack('>I', rows)[0]
        cols = images.read(4)
        cols = unpack('>I', cols)[0]

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

        print("MNIST Loaded with a cut size of: %i" % cut_size)
        if cut_size == 0:
            return X, y
        else:
            return X[:cut_size], y[:cut_size]
