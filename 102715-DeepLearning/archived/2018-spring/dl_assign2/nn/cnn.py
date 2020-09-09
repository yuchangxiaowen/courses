from __future__ import division, print_function
from .layers import Conv, Relu, MaxPool, Linear
from .loss import SoftmaxCE


class CNN(object):
    """Convolutional neural network with the following structures:
        conv + relu + pooling + linear + relu + linear + softmax
    """
    def __init__(self, image_size=(3, 32, 32), channels=32, conv_kernel=7,
                 pool_kernel=2, hidden_units=100, n_classes=10):
        """
        :param image_size: an 3 * H * W image, for color image,

        :param channels: channels in the convolution layer
        :param conv_kernel: kernel size of convolutional layer
        :param pool_kernel: kernel size of pooling layer
        :param hidden_units: number of hidden units in linear transform
        """

        # TODO: initialize the neural network. Define the layers
        # Your code should proceed like this:
        #
        # self.conv = Conv(32, 32)
        # self.relu = Relu()
        # ...
        if isinstance(conv_kernel, int):
            conv_height = conv_width = conv_kernel
        elif isinstance(conv_kernel, tuple) and len(conv_kernel) == 2:
            conv_height, conv_width = conv_kernel
        else:
            raise ValueError("Illegal conv conv kernel size!")
        if isinstance(pool_kernel, int):
            pool_height = pool_width = pool_kernel
        elif isinstance(pool_kernel, tuple) and len(pool_kernel) == 2:
            pool_height, pool_width = pool_kernel
        else:
            raise ValueError("Illegal conv pool kernel size!")

        self.conv = Conv(image_size[0], channels, conv_height, conv_width)
        self.relu = Relu()
        self.pooling = MaxPool(pool_kernel)
        after_conv_height = (image_size[1] - conv_height) // 1 + 1
        after_pool_height = (after_conv_height - pool_height) // 2 + 1
        after_conv_width = (image_size[2] - conv_width) // 1 + 1
        after_pool_width = (after_conv_width - pool_width) // 2 + 1
        self.fcn1 = Linear(channels * after_pool_width * after_pool_height,
                           hidden_units)
        self.fcn2 = Linear(hidden_units, n_classes)

        # TODO: Add the layers' parameters to the network, which will be assigned to optimizers

        self.param_groups = tuple((
            self.conv.params,
            self.fcn1.params,
            self.fcn2.params
        ))

    def oracle(self, x, y):
        """
        Oracle function to compute value of loss, score and gradient
        :param x: n * c * h * w tensor
        :param y: class label
        :return fx: loss value
        :return s: the output score of each class, this is the output of final linear layer.
        :return dout: gradient for each layer.
        """

        # TODO: Forward propagation
        # In addition to writing the output, you should also receive partial gradient with input of each layer,
        # this will be used in the backpropagation as well.

        dout = dict()  # a dictionary to receive the partial gradient
        out = dict()

        fx = None  # hold the loss value
        s = None  # hold the score of each class

        N = x.shape[0]

        out["conv"] = self.conv(x)
        out["relu1"] = self.relu(out["conv"])
        out["pooling"] = self.pooling(out["relu1"])
        out["pooling_flatten"] = out["pooling"].reshape((N, -1))
        out["fcn1"] = self.fcn1(out["pooling_flatten"])
        out["relu2"] = self.relu(out["fcn1"])
        out["fcn2"] = self.fcn2(out["relu2"])
        s = out["fcn2"]
        fx, dout["softmaxCE"] = SoftmaxCE()(s, y)

        # TODO: Backward propagation
        dout["fcn2"] = self.fcn2.backward(dout["softmaxCE"], out["relu2"])
        dout["relu2"] = self.relu.backward(dout["fcn2"], out["fcn1"])
        dout["fcn1"] = self.fcn1.backward(dout["relu2"], out["pooling_flatten"])
        dout["fcn1_4d"] = dout["fcn1"].reshape(out["pooling"].shape)
        dout["pooling"] = self.pooling.backward(dout["fcn1_4d"], out["relu1"])
        dout["relu1"] = self.relu.backward(dout["pooling"], out["conv"])
        dout["conv"] = self.conv.backward(dout["relu1"], x)

        return fx, s, dout

    def score(self, x):
        """
        Score of prediction, a seperate score function is needed in addition to the oracle. It is useful when checking
        accuracy.
        :param x: input features
        :return s: the output score of each class, this is the output of final linear layer.
        """
        # TODO: write the score function

        N = x.shape[0]

        s = self.fcn2.forward(self.relu.forward(self.fcn1.forward(
            self.pooling.forward(self.relu.forward(
                self.conv.forward(x))).reshape((N, -1)))))

        return s
