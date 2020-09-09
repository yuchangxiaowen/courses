from __future__ import division, print_function, absolute_import
import numpy as np
from nn.optimizer import SGD, Adam
from nn.utils import accuracy
from dataset import get_cifar10_data
from nn.cnn import CNN
from nn.layers import Conv_numba
from copy import deepcopy
from tqdm import tqdm
import matplotlib.pyplot as plt


def train_sgd(model, X_train, y_train, X_val, y_val, batch_size, n_epochs,
              lr=1e-2, lr_decay=0.8, momentum=0.):
    n_train = X_train.shape[0]
    iterations_per_epoch = max(n_train // batch_size, 1)

    loss_hist = []
    acc_hist = []

    # Define optimizer and set parameters
    opt_params = {'lr': lr, "momentum": momentum}
    sgd = SGD(model.param_groups, **opt_params)

    for epoch in tqdm(range(n_epochs)):
        inner_losses = []
        inner_accs = []
        for t in range(iterations_per_epoch):
            batch_mask = np.random.choice(n_train, batch_size)
            X_batch = X_train[batch_mask]
            y_batch = y_train[batch_mask]
            # Evaluate function value and gradient
            loss, score, _ = model.oracle(X_batch, y_batch)

            inner_losses.append(loss)

            # Perform stochastic gradient descent
            sgd.step()

            inner_accs.append(accuracy(score, y_batch))

            # print('(Iteration %d / %d, epoch %d) loss: %f, accu: %f' % (
            #     t + 1, iterations_per_epoch, epoch,
            #     inner_losses[-1], inner_accs[-1]))

            # At the end of every epoch, adjust the learning rate.
            epoch_end = (t + 1) % iterations_per_epoch == 0
            if epoch_end:
                epoch += 1
                opt_params['lr'] *= lr_decay
        loss_hist.append(sum(inner_losses) / len(inner_losses))
        acc_hist.append(sum(inner_accs) / len(inner_accs))

    val_loss, score, _ = model.oracle(X_val, y_val)
    val_acc = accuracy(score, y_val)
    return dict(loss_hist=loss_hist, acc_hist=acc_hist,
                val_loss=val_loss, val_acc=val_acc)


def train_adam(model, X_train, y_train, X_val, y_val, batch_size, n_epochs,
               lr=1e-3, beta1=.9, beta2=.999, eps=1e-8):
    n_train = X_train.shape[0]
    iterations_per_epoch = max(n_train // batch_size, 1)

    loss_hist = []
    acc_hist = []

    # Define optimizer and set parameters
    opt_params = {'lr': lr, "beta1": beta1, "beta2": beta2, "eps": eps}
    sgd = Adam(model.param_groups, **opt_params)

    for epoch in tqdm(range(n_epochs)):
        inner_losses = []
        inner_accs = []
        for t in range(iterations_per_epoch):
            batch_mask = np.random.choice(n_train, batch_size)
            X_batch = X_train[batch_mask]
            y_batch = y_train[batch_mask]
            # Evaluate function value and gradient
            loss, score, _ = model.oracle(X_batch, y_batch)

            inner_losses.append(loss)

            # Perform stochastic gradient descent
            sgd.step()

            inner_accs.append(accuracy(score, y_batch))

        loss_hist.append(sum(inner_losses) / len(inner_losses))
        acc_hist.append(sum(inner_accs) / len(inner_accs))

    val_loss, score, _ = model.oracle(X_val, y_val)
    val_acc = accuracy(score, y_val)
    return dict(loss_hist=loss_hist, acc_hist=acc_hist,
                val_loss=val_loss, val_acc=val_acc)


def prepare_model(cache=None):
    model = CNN()
    model.conv = Conv_numba(3, 32, 7, 7)
    model.param_groups = tuple((
        model.conv.params,
        model.fcn1.params,
        model.fcn2.params
    ))
    if cache is not None:
        for model_group, cache_group in zip(model.param_groups, cache):
            for k, v in model.param_groups:
                model_group[k]["param"] = cache_group[k]["param"].copy()

    return model


def fetch_data():
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

    return X_train, y_train, X_val, y_val


def show_performance(progress):
    _, ax = plt.subplots()
    for name, result in progress.items():
        ax.plot(result["loss_hist"], label=name)
    ax.set_xlabel('iteration')
    ax.set_ylabel('loss')
    ax.set_title('Loss versus iteration')
    ax.legend(loc="best")

    _, ax = plt.subplots()
    for name, result in progress.items():
        ax.plot(result["acc_hist"], label=name)
    ax.set_xlabel('iteration')
    ax.set_ylabel('accuarcy')
    ax.set_title('Accuarcy versus iteration')
    ax.legend(loc="best")

    plt.show()

    for name, result in progress.items():
        print("Optimizer: {}, val loss = {}, val acc = {}".format(
            name, result["val_loss"], result["val_acc"]))


if __name__ == '__main__':
    np.random.seed(1000)
    X_train, y_train, X_val, y_val = fetch_data()
    progress = dict()

    model = prepare_model()
    cache = deepcopy(model.param_groups)
    progress["sgd"] = train_sgd(
        model, X_train, y_train, X_val, y_val,
        batch_size=50, n_epochs=50)

    model = prepare_model(cache)
    progress["momentum"] = train_sgd(
        model, X_train, y_train, X_val, y_val,
        batch_size=50, n_epochs=50, momentum=.99)

    model = prepare_model(cache)
    progress["adam"] = train_adam(
        model, X_train, y_train, X_val, y_val,
        batch_size=50, n_epochs=50)

    show_performance(progress)
