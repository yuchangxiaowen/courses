
import numpy as np


def softmax(x):
    x_bar = x - np.max(x, axis=1, keepdims=True)
    z = np.sum(np.exp(x_bar), axis=1, keepdims=True)
    return np.exp(x_bar) / z


class SoftmaxCE(object):
    """
    Compute the cross entropy loss with softmax transform.
    """

    def __init__(self):
        pass

    @staticmethod
    def __call__(x, y):
        """
        :param x: n_samples by n_features matrix
        :param y:
        :return: loss and gradient
        """

        # enhance numerical stability
        sf = softmax(x)
        n = x.shape[0]
        sf_log = -np.log(np.maximum(sf[range(n), y], 1e-8))
        loss = np.mean(sf_log)
        g = sf
        g[range(n), y] -= 1
        g /= n
        return loss, g
