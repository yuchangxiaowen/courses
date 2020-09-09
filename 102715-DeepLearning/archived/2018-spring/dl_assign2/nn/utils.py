
import numpy as np


def accuracy(score, y):
    r"""
    Compute the classification accuracy.
    :param score: n by n_classes score matrix
    :param y: n by 1 vector of data label
    :return accu: a scalar in [0, 1], accuracy

    """
    acc = np.mean(score.argmax(axis=1) == y)
    return acc
