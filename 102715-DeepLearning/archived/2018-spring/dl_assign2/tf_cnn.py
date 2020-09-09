from __future__ import division, print_function, absolute_import
from dataset import get_cifar10_data
import tensorflow as tf
import numpy as np
import argparse
from tqdm import tqdm


def train_small():
    tf.reset_default_graph()

    with tf.device("/gpu:0"):
        data = get_cifar10_data()
        num_train = 100
        data = {
            'X_train': data['X_train'][:num_train],
            'y_train': data['y_train'][:num_train],
            'X_val': data['X_val'],
            'y_val': data['y_val'],
        }
        X_train, y_train, X_val, y_val = \
            data['X_train'], data['y_train'].astype(np.int32), \
            data['X_val'], data['y_val'].astype(np.int64)

        inputs = tf.placeholder(tf.float32, shape=(None, 3, 32, 32))
        labels = tf.placeholder(tf.int64, shape=(None, ))

        conv = tf.layers.conv2d(inputs, filters=32, kernel_size=7,
                                data_format='channels_first',
                                activation=tf.nn.relu)
        pool = tf.layers.max_pooling2d(
            inputs=conv, pool_size=[2, 2], strides=2)
        flat = tf.reshape(pool, [-1, 13 * 13 * 32])
        dense = tf.layers.dense(inputs=flat, units=100, activation=tf.nn.relu)
        logits = tf.layers.dense(inputs=dense, units=10)
        loss = tf.losses.sparse_softmax_cross_entropy(
            labels=labels, logits=logits)
        optimizer = tf.train.GradientDescentOptimizer(learning_rate=1e-3)
        train_op = optimizer.minimize(loss)

        correct_prediction = tf.equal(tf.argmax(logits, 1), labels)
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

        batch_size = 50
        epoch = 50
        iterations_per_epoch = max(num_train // batch_size, 1)

        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            for i in range(epoch):
                for t in range(iterations_per_epoch):
                    batch_mask = np.random.choice(num_train, batch_size)
                    X_batch = X_train[batch_mask]
                    y_batch = y_train[batch_mask]
                    # Evaluate function value and gradient
                    train_loss, _, train_acc = sess.run(
                        [loss, train_op, accuracy],
                        feed_dict={inputs: X_batch, labels: y_batch})

            val_loss, val_acc = sess.run(
                [loss, accuracy],
                feed_dict={inputs: X_val, labels: y_val})
            print('Validation loss: %f, accu: %f' % (val_loss, val_acc))


def train_raw():
    tf.reset_default_graph()

    with tf.device("/gpu:0"):
        data = get_cifar10_data()
        num_train = data['X_train'].shape[0]
        X_train, y_train, X_val, y_val, X_test, y_test = \
            data['X_train'], data['y_train'].astype(np.int32), \
            data['X_val'], data['y_val'].astype(np.int64), \
            data['X_test'], data['y_test'].astype(np.int64)

        inputs = tf.placeholder(tf.float32, shape=(None, 3, 32, 32))
        labels = tf.placeholder(tf.int64, shape=(None, ))

        conv = tf.layers.conv2d(inputs, filters=32, kernel_size=7,
                                data_format='channels_first',
                                activation=tf.nn.relu)
        pool = tf.layers.max_pooling2d(
            inputs=conv, pool_size=[2, 2], strides=2)
        flat = tf.reshape(pool, [-1, 13 * 13 * 32])
        dense = tf.layers.dense(inputs=flat, units=100, activation=tf.nn.relu)
        logits = tf.layers.dense(inputs=dense, units=10)
        loss = tf.losses.sparse_softmax_cross_entropy(
            labels=labels, logits=logits)

        train_summary = []
        val_summary = []

        train_summary.append(tf.summary.scalar("train_loss", loss))
        val_summary.append(tf.summary.scalar("val_loss", loss))
        optimizer = tf.train.AdamOptimizer()
        train_op = optimizer.minimize(loss)

        correct_prediction = tf.equal(tf.argmax(logits, 1), labels)
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
        train_summary.append(tf.summary.scalar("train_acc", accuracy))
        val_summary.append(tf.summary.scalar("val_acc", accuracy))

        train_summary = tf.summary.merge(train_summary)
        val_summary = tf.summary.merge(val_summary)

        batch_size = 120
        epoch = 1000
        iterations_per_epoch = max(num_train // batch_size, 1)

        writer = tf.summary.FileWriter("tmp/raw")

        config = tf.ConfigProto(allow_soft_placement=True)
        with tf.Session(config=config) as sess:
            sess.run(tf.global_variables_initializer())
            for i in tqdm(range(epoch)):
                for t in range(iterations_per_epoch):
                    batch_mask = np.random.choice(num_train, batch_size)
                    X_batch = X_train[batch_mask]
                    y_batch = y_train[batch_mask]
                    # Evaluate function value and gradient
                    _, s = sess.run(
                        [train_op, train_summary],
                        feed_dict={inputs: X_batch, labels: y_batch})
                    writer.add_summary(s, i * epoch + t)
                s = sess.run(val_summary,
                             feed_dict={inputs: X_val, labels: y_val})
                writer.add_summary(s, i)

            test_loss, test_acc = sess.run(
                [loss, accuracy],
                feed_dict={inputs: X_test, labels: y_test})
            print('Test loss: %f, accu: %f' % (test_loss, test_acc))


def train_refined_2conv():
    tf.reset_default_graph()

    with tf.device("/gpu:0"):
        data = get_cifar10_data()
        num_train = data['X_train'].shape[0]

        X_train, y_train, X_val, y_val, X_test, y_test = \
            data['X_train'], data['y_train'].astype(np.int32), \
            data['X_val'], data['y_val'].astype(np.int64), \
            data['X_test'], data['y_test'].astype(np.int64)

        F1, F2, kernel_sz = 32, 64, 7

        inputs = tf.placeholder(tf.float32, shape=(None, 3, 32, 32))
        labels = tf.placeholder(tf.int64, shape=(None, ))
        istraining = tf.placeholder(tf.bool)

        conv1 = tf.layers.conv2d(inputs, filters=F1, kernel_size=kernel_sz,
                                 data_format='channels_first',
                                 activation=tf.nn.relu)
        bn1 = tf.layers.batch_normalization(inputs=conv1, training=istraining)
        pool1 = tf.layers.max_pooling2d(
            inputs=bn1, pool_size=[2, 2], strides=2,
            data_format='channels_first')
        conv2 = tf.layers.conv2d(pool1, filters=F2, kernel_size=kernel_sz,
                                 data_format='channels_first',
                                 activation=tf.nn.relu)
        bn2 = tf.layers.batch_normalization(inputs=conv2, training=istraining)
        pool2 = tf.layers.max_pooling2d(
            inputs=bn2, pool_size=[2, 2], strides=2,
            data_format='channels_first')

        height = width = ((32 - kernel_sz + 1) // 2 - kernel_sz + 1) // 2
        flat = tf.reshape(pool2, [-1, height * width * F2])
        logits = tf.layers.dense(inputs=flat, units=10)

        loss = tf.losses.sparse_softmax_cross_entropy(
            labels=labels, logits=logits)
        optimizer = tf.train.AdamOptimizer()
        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        with tf.control_dependencies(update_ops):
            train_op = optimizer.minimize(loss)

        train_summary = []
        val_summary = []

        train_summary.append(tf.summary.scalar("train_loss", loss))
        val_summary.append(tf.summary.scalar("val_loss", loss))

        correct_prediction = tf.equal(tf.argmax(logits, 1), labels)
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

        train_summary.append(tf.summary.scalar("train_acc", accuracy))
        val_summary.append(tf.summary.scalar("val_acc", accuracy))

        train_summary = tf.summary.merge(train_summary)
        val_summary = tf.summary.merge(val_summary)

        batch_size = 120
        epoch = 1000
        iterations_per_epoch = max(num_train // batch_size, 1)

        writer = tf.summary.FileWriter("tmp/2conv")

        config = tf.ConfigProto(allow_soft_placement=True)
        with tf.Session(config=config) as sess:
            sess.run(tf.global_variables_initializer())
            for i in tqdm(range(epoch)):
                for t in range(iterations_per_epoch):
                    batch_mask = np.random.choice(num_train, batch_size)
                    X_batch = X_train[batch_mask]
                    y_batch = y_train[batch_mask]
                    # Evaluate function value and gradient
                    sess.run(train_op, feed_dict={inputs: X_batch,
                                                  labels: y_batch,
                                                  istraining: True})
                    s = sess.run(
                        train_summary,
                        feed_dict={inputs: X_batch,
                                   labels: y_batch,
                                   istraining: False})
                    writer.add_summary(s, i * epoch + t)
                s = sess.run(
                    val_summary,
                    feed_dict={inputs: X_val,
                               labels: y_val,
                               istraining: False})
                writer.add_summary(s, i)

            test_loss, test_acc = sess.run(
                [loss, accuracy],
                feed_dict={inputs: X_test, labels: y_test, istraining: False})
            print('Test loss: %f, accu: %f' % (test_loss, test_acc))


def train_refined_2linear():
    tf.reset_default_graph()

    with tf.device("/gpu:0"):
        data = get_cifar10_data()
        num_train = data['X_train'].shape[0]
        X_train, y_train, X_val, y_val, X_test, y_test = \
            data['X_train'], data['y_train'].astype(np.int32), \
            data['X_val'], data['y_val'].astype(np.int64), \
            data['X_test'], data['y_test'].astype(np.int64)

        inputs = tf.placeholder(tf.float32, shape=(None, 3, 32, 32))
        labels = tf.placeholder(tf.int64, shape=(None, ))
        istraining = tf.placeholder(tf.bool)

        conv = tf.layers.conv2d(inputs, filters=32, kernel_size=7,
                                data_format='channels_first',
                                activation=tf.nn.relu)
        bn = tf.layers.batch_normalization(inputs=conv, training=istraining)
        pool = tf.layers.max_pooling2d(
            inputs=bn, pool_size=[2, 2], strides=2)
        flat = tf.reshape(pool, [-1, 13 * 13 * 32])
        dense = tf.layers.dense(inputs=flat, units=500, activation=tf.nn.relu)
        dropout = tf.layers.dropout(dense, training=istraining)
        logits = tf.layers.dense(inputs=dropout, units=10)
        loss = tf.losses.sparse_softmax_cross_entropy(
            labels=labels, logits=logits)

        train_summary = []
        val_summary = []

        train_summary.append(tf.summary.scalar("train_loss", loss))
        val_summary.append(tf.summary.scalar("val_loss", loss))
        optimizer = tf.train.AdamOptimizer()
        train_op = optimizer.minimize(loss)

        correct_prediction = tf.equal(tf.argmax(logits, 1), labels)
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
        train_summary.append(tf.summary.scalar("train_acc", accuracy))
        val_summary.append(tf.summary.scalar("val_acc", accuracy))

        train_summary = tf.summary.merge(train_summary)
        val_summary = tf.summary.merge(val_summary)

        batch_size = 120
        epoch = 1000
        iterations_per_epoch = max(num_train // batch_size, 1)

        writer = tf.summary.FileWriter("tmp/2linear")

        config = tf.ConfigProto(allow_soft_placement=True)
        with tf.Session(config=config) as sess:
            sess.run(tf.global_variables_initializer())
            for i in tqdm(range(epoch)):
                for t in range(iterations_per_epoch):
                    batch_mask = np.random.choice(num_train, batch_size)
                    X_batch = X_train[batch_mask]
                    y_batch = y_train[batch_mask]
                    # Evaluate function value and gradient
                    sess.run(train_op, feed_dict={inputs: X_batch,
                                                  labels: y_batch,
                                                  istraining: True})
                    s = sess.run(
                        train_summary,
                        feed_dict={inputs: X_batch,
                                   labels: y_batch,
                                   istraining: False})
                    writer.add_summary(s, i * epoch + t)
                s = sess.run(
                    val_summary,
                    feed_dict={inputs: X_val,
                               labels: y_val,
                               istraining: False})
                writer.add_summary(s, i)

            test_loss, test_acc = sess.run(
                [loss, accuracy],
                feed_dict={inputs: X_test, labels: y_test, istraining: False})
            print('Test loss: %f, accu: %f' % (test_loss, test_acc))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config',
                        choices=['small', 'raw', '2conv', '2linear'],
                        default='raw')
    args = parser.parse_args()

    if args.config == "small":
        train_small()
    elif args.config == "raw":
        train_raw()
    elif args.config == "2conv":
        train_refined_2conv()
    elif args.config == "2linear":
        train_refined_2linear()
    else:
        raise ValueError("Unrecognized config value.")
