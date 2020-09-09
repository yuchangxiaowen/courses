"""
This file contains various layers in neural network architecture.
"""

import abc
import numpy as np

from numba import float32, int32, jit
from numba.types import Tuple

class Layer(object):
    """
    Base class for neural network layer, this class contains abstract methods, hence should not be instantiated.
    Args:
        params: A dictionary of parameters,
                params[k] == v, where k is the name string of the parameter, and v is a dictionary
                containing its properties:
                    v['param']: parameter value
                    v['grad']: gradient
    """

    def __init__(self):
        self.params = dict()

    @abc.abstractmethod
    def forward(self, x):
        r"""Evaluate input features and return output
        :param x: input features
        :return f(x): output features
        """
        pass

    @abc.abstractmethod
    def backward(self, grad_in, x):
        r"""Compute gradient and backpropagate it. The updated gradient should be stored in the field of self.params for
            future reference.
        :param grad_in: gradient from back propagation
        :param x: input features
        :return grad_x: gradient propagated backward to the next layer, namely gradient w.r.t. x
        """
        pass

    # Make the Layer type callable
    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)


class Conv(Layer):
    """Convolutional layer
    Args:
        w: convolutional filter:
        b: bias
    """

    def __init__(self, in_channels, out_channels, height, width, stride=1, padding=0, init_scale=1e-2):
        super(Conv, self).__init__()
        # TODO: initial the parameter and value

        self.params['w'] = dict(param=np.random.randn(
            out_channels, in_channels, height, width) * init_scale, grad=None)
        self.params['b'] = dict(param=np.zeros(out_channels), grad=None)
        self.stride = stride
        self.padding = padding

    def forward(self, x):
        r"""
        :param x: a 4-d tensor, N_samples by n_Channels by Height by Width
        :return: output of convolutional kernel
        """
        # TODO
        out = None

        w, b = self.params['w']["param"], self.params['b']["param"]
        H, W, HH, WW, pad, stride = \
            x.shape[2], x.shape[3], w.shape[2], w.shape[3], \
            self.padding, self.stride
        newH, newW = \
            1 + (H + 2 * pad - HH) // stride, \
            1 + (W + 2 * pad - WW) // stride

        x_padded = np.pad(x, ((0, ), (0, ), (pad, ), (pad, )),
                          "constant", constant_values=0)
        rows = []
        for i in range(newH):
            cols = []
            for j in range(newW):
                cols.append(np.transpose(np.array(list(map(
                    lambda yy: (x_padded[:, :,
                                         i * stride: i * stride + HH,
                                         j * stride: j * stride + WW] *
                                np.squeeze(yy)
                                ).sum(axis=(1, 2, 3)),
                    np.split(w, w.shape[0]))))))
            rows.append(np.stack(cols, axis=-1))
        out = np.stack(rows, axis=2) + b.reshape((1, -1, 1, 1))

        return out

    def backward(self, grad_in, x):

        # TODO
        grad_x = None

        dw, db = None, None
        w = self.params['w']["param"]
        H, W, HH, WW, pad, stride, F, C, N = \
            x.shape[2], x.shape[3], w.shape[2], w.shape[3], \
            self.padding, self.stride, \
            w.shape[0], w.shape[1], x.shape[0]
        newH, newW = \
            1 + (H + 2 * pad - HH) // stride, \
            1 + (W + 2 * pad - WW) // stride

        db = grad_in.sum(axis=(0, 2, 3))

        x_padded = np.pad(x, ((0, ), (0, ), (pad, ), (pad, )),
                          "constant", constant_values=0)
        dw = np.zeros((F, C, HH, WW))
        grad_x_padded = np.zeros((N, C, H + 2 * pad, W + 2 * pad))
        for i in range(newH):
            for j in range(newW):
                dw += np.array(list(map(
                    lambda yy: (np.transpose(x_padded[
                        :, :,
                        i * stride: i * stride + HH,
                        j * stride: j * stride + WW], (1, 0, 2, 3)) *
                        np.squeeze(yy, axis=0)
                    ).sum(axis=1),
                    np.split(np.transpose(grad_in[:, :, i: i + 1, j: j + 1],
                                          (1, 0, 2, 3)), F))))

                grad_x_padded[:, :,
                              i * stride: i * stride + HH,
                              j * stride: j * stride + WW] += \
                    np.array(list(map(
                        lambda yy: (np.transpose(w, (1, 0, 2, 3)) *
                                    np.squeeze(yy, axis=0)).sum(axis=1),
                        np.split(grad_in[:, :, i: i + 1, j: j + 1], N))))
        grad_x = grad_x_padded if pad == 0 \
            else grad_x_padded[:, :, pad: -pad, pad: -pad]

        self.params["b"]["grad"] = db
        self.params["w"]["grad"] = dw
        return grad_x


class Linear(Layer):
    r"""
    Apply linear transform to features

    Args:
        w: n_in by n_out ndarray
        b: 1 by n_out ndarray
    """

    def __init__(self, in_features, out_features, init_scale=1e-2):

        super(Linear, self).__init__()
        self.params['w'] = dict(param=np.random.randn(
            in_features, out_features) * init_scale, grad=None)
        self.params['b'] = dict(param=np.zeros(out_features), grad=None)

    def forward(self, x):
        """
        :param x: input features of dimension [n, d1, d2,..., dm]
        :return: output features
        """

        # TODO: write forward propagation

        out = x.dot(self.params["w"]["param"]) + \
            self.params["b"]["param"]

        return out

    def backward(self, grad_in, x):
        """
        Backward propagation of linear layer
        """

        # TODO: write backward propagation

        grad_x = grad_in.dot(self.params["w"]["param"].T)
        self.params["w"]["grad"] = x.T.dot(grad_in)
        self.params["b"]["grad"] = grad_in.sum(axis=0)

        return grad_x


class Relu(Layer):

    def __init__(self):
        super(Relu, self).__init__()

    def forward(self, x):

        return np.maximum(x, 0)

    def backward(self, grad_in, x):

        return grad_in * (x > 0)


class MaxPool(Layer):
    """Max pooling
    """

    def __init__(self, kernel_size, stride=2, padding=0):
        super(MaxPool, self).__init__()
        # TODO: initialize pooling layer
        if isinstance(kernel_size, int):
            kernel_size = (kernel_size, kernel_size)
        elif not isinstance(kernel_size, tuple):
            raise ValueError("Illegal type of kernel_size!")
        self.pool_height = kernel_size[0]
        self.pool_width = kernel_size[1]
        self.stride = stride
        self.padding = padding

    def forward(self, x):

        # TODO: write forward propagation

        out = None

        N, C, H, W = x.shape
        HH, WW, stride, pad = \
            self.pool_height, \
            self.pool_width, \
            self.stride, \
            self.padding
        newH, newW = \
            1 + (H - HH + 2 * pad) // stride, \
            1 + (W - WW + 2 * pad) // stride
        x_padded = np.pad(x, ((0, ), (0, ), (pad, ), (pad, )),
                          "constant", constant_values=0)
        out = np.empty((N, C, newH, newW), dtype=x.dtype)
        for i in range(newH):
            for j in range(newW):
                out[:, :, i, j] = x_padded[
                    :, :,
                    i * stride: i * stride + HH,
                    j * stride: j * stride + WW].max(axis=(2, 3))

        return out

    def backward(self, grad_in, x):

        # TODO: write backward propagation

        grad_x = None

        N, C, H, W = x.shape
        HH, WW, stride, pad = \
            self.pool_height, \
            self.pool_width, \
            self.stride, \
            self.padding
        newH, newW = \
            1 + (H - HH + 2 * pad) // stride, \
            1 + (W - WW + 2 * pad) // stride
        x_padded = np.pad(x, ((0, ), (0, ), (pad, ), (pad, )),
                          "constant", constant_values=0)
        grad_x = np.zeros_like(x_padded)
        for i in range(newH):
            for j in range(newW):
                local_flat = x_padded[
                    :, :,
                    i * stride: i * stride + HH,
                    j * stride: j * stride + WW].reshape(N, C, -1)
                h_idx, w_idx = np.unravel_index(np.argmax(
                    local_flat, axis=-1), (HH, WW))
                grad_x[np.repeat(np.arange(N), C),
                       np.tile(np.arange(C), N),
                       i * stride + h_idx.ravel(),
                       j * stride + w_idx.ravel()] += \
                    grad_in[:, :, i, j].ravel()
        if pad != 0:
            grad_x = grad_x[:, :, pad: -pad, pad: -pad]

        return grad_x


class Conv_numba(Conv):

    def forward(self, x):

        out, x_cols = forward_numba(
            x, self.params['w']["param"],
            self.params['b']["param"],
            self.padding, self.stride)

        self.x_cols = x_cols
        return out

    def backward(self, grad_in, x):

        grad_x, dw, db = backward_numba(
            grad_in, x, self.params['w']["param"],
            self.padding, self.stride, self.x_cols)
        self.params["b"]["grad"] = db
        self.params["w"]["grad"] = dw
        return grad_x


@jit(Tuple((float32[:, :, :, :], float32[:, :]))(float32[:, :, :, :], float32[:, :, :, :], float32[:], int32, int32), nogil=True, cache=True)
def forward_numba(x, w, b, pad, stride):
    N, C, H, W = x.shape
    F, _, HH, WW = w.shape
    newH, newW = \
        1 + (H + 2 * pad - HH) // stride, \
        1 + (W + 2 * pad - WW) // stride

    if pad == 0:
        x_padded = x
    else:
        shape = x.shape
        shape[2] += 2 * pad
        shape[3] += 2 * pad
        x_padded = np.zeros(shape)
        x_padded[:, :, pad: -pad, pad: -pad] = x

    x_cols = np.zeros((C * HH * WW, N * newH * newW))

    for c in range(C):
        for yy in range(newH):
            for xx in range(newW):
                for ii in range(HH):
                    for jj in range(WW):
                        row = c * WW * HH + ii * HH + jj
                        for i in range(N):
                            col = yy * newW * N + xx * N + i
                            x_cols[row, col] = x_padded[i, c, stride * yy + ii, stride * xx + jj]

    res = w.reshape((F, -1)).dot(x_cols) + b.reshape(-1, 1)

    out = res.reshape(F, newH, newW, N)
    out = out.transpose(3, 0, 1, 2)
    return out, x_cols


@jit(Tuple((float32[:, :, :, :], float32[:, :, :, :], float32[:]))(float32[:, :, :, :], float32[:, :, :, :], float32[:, :, :, :], int32, int32, float32[:, :]), nogil=True, cache=True)
def backward_numba(grad_in, x, w, pad, stride, x_cols):
    N, C, H, W = x.shape
    F, _, HH, WW = w.shape
    newH, newW = \
        1 + (H + 2 * pad - HH) // stride, \
        1 + (W + 2 * pad - WW) // stride

    if pad == 0:
        x_padded = x
    else:
        shape = x.shape
        shape[2] += 2 * pad
        shape[3] += 2 * pad
        x_padded = np.zeros(shape)
        x_padded[:, :, pad: -pad, pad: -pad] = x

    db = grad_in.sum(axis=(0, 2, 3))

    grad_in_reshaped = grad_in.transpose(1, 2, 3, 0).reshape(F, -1)
    dw = grad_in_reshaped.dot(x_cols.T).reshape(w.shape)

    dx_cols = w.reshape(F, -1).T.dot(grad_in_reshaped)
    grad_x = np.zeros((N, C, H + 2 * pad, W + 2 * pad))

    for c in range(C):
        for ii in range(HH):
            for jj in range(WW):
                row = c * WW * HH + ii * HH + jj
                for yy in range(newH):
                    for xx in range(newW):
                        for i in range(N):
                            col = yy * newW * N + xx * N + i
                            grad_x[i, c, stride * yy + ii, stride * xx + jj] += dx_cols[row, col]

    if pad > 0:
        grad_x = grad_x[:, :, pad:-pad, pad:-pad]

    return grad_x, dw, db
