from __future__ import division, print_function, absolute_import
import numpy as np
from nn.optimizer import SGD
from nn.utils import accuracy
from dataset import get_cifar10_data
from nn.cnn import CNN
from nn.layers import Conv_numba
import argparse


def train(model, X_train, y_train, X_val, y_val, batch_size, n_epochs, lr=1e-2,
          lr_decay=0.8, verbose=True, print_level=100):
    n_train = X_train.shape[0]
    iterations_per_epoch = max(n_train // batch_size, 1)

    loss_hist = []

    # Define optimizer and set parameters
    opt_params = {'lr': lr}
    sgd = SGD(model.param_groups, **opt_params)

    for epoch in range(n_epochs):
        for t in range(iterations_per_epoch):
            batch_mask = np.random.choice(n_train, batch_size)
            X_batch = X_train[batch_mask]
            y_batch = y_train[batch_mask]
            # Evaluate function value and gradient
            loss, score, _ = model.oracle(X_batch, y_batch)

            loss_hist.append(loss)

            # Perform stochastic gradient descent

            sgd.step()

            # Maybe print training loss
            if verbose and t % print_level == 0:
                train_acc = accuracy(score, y_batch)
                print('(Iteration %d / %d, epoch %d) loss: %f, accu: %f' % (
                    t + 1, iterations_per_epoch, epoch + 1,
                    loss_hist[-1], train_acc))

            # At the end of every epoch, adjust the learning rate.
            epoch_end = (t + 1) % iterations_per_epoch == 0
            if epoch_end:
                epoch += 1
                opt_params['lr'] *= lr_decay

    val_loss, score, _ = model.oracle(X_val, y_val)
    val_acc = accuracy(score, y_val)
    print('Validation loss: %f, accu: %f' % (
        val_loss, val_acc))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--numba', dest='numba', action='store_true')
    parser.add_argument('--no-numba', dest='numba', action='store_false')
    parser.set_defaults(numba=False)

    parser.add_argument('--verbose', dest='verbose', action='store_true')
    parser.add_argument('--no-verbose', dest='verbose', action='store_false')
    parser.set_defaults(verbose=True)

    args = parser.parse_args()

    np.random.seed(1000)
    data = get_cifar10_data()
    num_train = 100
    data = {
        'X_train': data['X_train'][:num_train],
        'y_train': data['y_train'][:num_train],
        'X_val': data['X_val'],
        'y_val': data['y_val'],
    }
    X_train, y_train, X_val, y_val = \
        data['X_train'], data['y_train'], data['X_val'], data['y_val']

    model = CNN()
    if args.numba:
        model.conv = Conv_numba(3, 32, 7, 7)
        model.param_groups = tuple((
            model.conv.params,
            model.fcn1.params,
            model.fcn2.params
        ))

    train(model, X_train, y_train, X_val, y_val,
          batch_size=50, n_epochs=50, print_level=1, verbose=args.verbose)
