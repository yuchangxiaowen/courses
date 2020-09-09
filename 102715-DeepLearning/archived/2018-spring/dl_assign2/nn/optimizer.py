
"""
This file consists of classes of first order algorithms for neural network optimization.
The base class optimized
"""

import abc
import numpy as np


class Optimizer(object):
    r"""Base class for optimizers
    Args:
        param_groups: a list of all the parameters in the neural network models
        configs: optimization hyper-parameters
    """

    def __init__(self, param_groups):
        self.param_groups = param_groups

    @abc.abstractmethod
    def step(self):
        pass


class SGD(Optimizer):
    """Stochastic gradient descent
    """

    def __init__(self, param_groups, lr=1e-2, weight_decay=0.0, momentum=0.0):

        super(SGD, self).__init__(param_groups)
        self.configs = dict(
            lr=lr,
            weight_decay=weight_decay,
            momentum=momentum)

    def step(self):

        lr = self.configs['lr']
        weight_decay = self.configs['weight_decay']
        momentum = self.configs['momentum']

        # TODO： add momentum term in the algorithm
        # For achieving this goal, you can add more fields in the param groups

        for group in self.param_groups:
            for k, p in group.items():
                x = p['param']
                g = p['grad']
                if momentum == 0:
                    if weight_decay > 0 and k == 'w':
                        x -= lr * (g + weight_decay * x)
                    else:
                        x -= lr * g
                else:
                    if "momen" in p and p["momen"] is not None:
                        v = p["momen"]
                    else:
                        v = p["momen"] = np.zeros_like(g)
                    if weight_decay > 0 and k == 'w':
                        v *= momentum
                        v -= lr * (g + weight_decay * x)
                    else:
                        v *= momentum
                        v -= lr * g
                    x += v


class Adam(Optimizer):
    """Stochastic gradient descent
    """

    def __init__(self, param_groups, lr=1e-3, beta1=.9, beta2=.999, eps=1e-8):

        super(Adam, self).__init__(param_groups)
        self.configs = dict(
            lr=lr,
            beta1=beta1,
            beta2=beta2,
            eps=eps)

    def step(self):

        lr = self.configs['lr']
        beta1 = self.configs['beta1']
        beta2 = self.configs['beta2']
        eps = self.configs['eps']

        # TODO： add momentum term in the algorithm
        # For achieving this goal, you can add more fields in the param groups

        for group in self.param_groups:
            for k, p in group.items():
                x = p['param']
                g = p['grad']

                if "momen" in p and p["momen"] is not None:
                    v = p["momen"]
                else:
                    v = p["momen"] = np.zeros_like(g)

                if "square_momen" in p and p["square_momen"] is not None:
                    m = p["square_momen"]
                else:
                    m = p["square_momen"] = np.zeros_like(g)

                m *= beta1
                m += (1 - beta1) * g
                v *= beta2
                v += (1 - beta2) * (g ** 2)

                x -= lr * m / (1 - beta1) / (eps + np.sqrt(v / (1 - beta2)))
