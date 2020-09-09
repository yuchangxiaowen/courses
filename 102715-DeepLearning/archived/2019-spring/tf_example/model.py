import numpy as np
import tensorflow as tf


def fc_layer(inputs, input_sz, output_sz,
             activation=True, name="fc"):
    epsilon = (6. / float(input_sz + output_sz)) ** .5
    with tf.name_scope(name):
        W = tf.get_variable(
            '{}/W'.format(name),
            [input_sz, output_sz],
            initializer=tf.random_uniform_initializer(
                minval=-epsilon, maxval=epsilon)
        )
        b = tf.get_variable('{}/b'.format(name), [output_sz],
            initializer=tf.zeros_initializer())

        if activation:
            outputs = tf.nn.relu(tf.matmul(inputs, W) + b)
        else:
            outputs = tf.matmul(inputs, W) + b

        tf.summary.histogram("weights", W)
        tf.summary.histogram("biases", b)
        if activation:
            tf.summary.histogram("activations", outputs)

        return outputs


class Model(object):

    def add_placeholders(self):
        self.input_ph = tf.placeholder(
            tf.float32,
            (None, self.cfg.n_features),
            name="inputs")
        self.labels_ph = tf.placeholder(
            tf.float32,
            (None, self.cfg.n_classes),
            name="labels")

    def add_prediction_op(self):
        fc1 = fc_layer(
            self.input_ph,
            self.cfg.n_features,
            self.cfg.hidden_sz,
            name="fc1")
        score = fc_layer(
            fc1,
            self.cfg.hidden_sz,
            self.cfg.n_classes,
            activation=False,
            name="fc2")
        return score

    def add_loss_op(self, pred):
        with tf.name_scope("loss"):
            with tf.name_scope("cross_entropy"):
                xent = tf.reduce_mean(
                    tf.nn.softmax_cross_entropy_with_logits(
                        labels=self.labels_ph,
                        logits=pred)
                )
            with tf.name_scope("l2_regularization"):
                l2s = []
                for name in ["fc1", "fc2"]:
                    with tf.variable_scope("", reuse=True):
                        l2s.append(tf.nn.l2_loss(
                            tf.get_variable("{}/W".format(name))))
                l2_loss = self.cfg.reg_strength * tf.add_n(l2s)
            loss = xent + l2_loss
        tf.summary.scalar("loss", loss)
        return loss

    def add_metric_op(self, pred):
        with tf.name_scope("accuracy"):
            correct_pred = tf.equal(
                tf.argmax(pred, 1),
                tf.argmax(self.labels_ph, 1))
            accuracy = tf.reduce_mean(
                tf.cast(correct_pred, tf.float32))
        tf.summary.scalar("accuracy", accuracy)
        return accuracy

    def add_training_op(self, loss):
        train_op = tf.train.AdamOptimizer(
            self.cfg.lr).minimize(loss)
        return train_op

    def build(self):
        self.add_placeholders()
        self.pred = self.add_prediction_op()
        self.loss = self.add_loss_op(self.pred)
        self.metric = self.add_metric_op(self.pred)
        self.summary_op = tf.summary.merge_all()
        self.train_op = self.add_training_op(self.loss)

    def create_feed_dict(self, inputs_batch, labels_batch=None):
        feed_dict = {
            self.input_ph: inputs_batch
        } if labels_batch is None else {
            self.input_ph: inputs_batch,
            self.labels_ph: labels_batch
        }
        return feed_dict

    def train_on_batch(self, sess, writer,
            inputs_batch, labels_batch):
        feed = self.create_feed_dict(
            inputs_batch,
            labels_batch=labels_batch)
        sess.run(self.train_op, feed_dict=feed)
        s = sess.run(self.summary_op, feed_dict=feed)
        writer.add_summary(s, self.iter)
        self.iter += 1

    def compute_dev_metric(self, sess, inputs, labels):
        feed = self.create_feed_dict(
            inputs,
            labels_batch=labels)
        metric = sess.run(self.metric, feed_dict=feed)
        return metric

    def run_epoch(self, sess, writer, train_set, dev_set):
        idx = np.random.permutation(train_set["X"].shape[0])
        n_batches = len(train_set["X"]) // self.cfg.batch_sz
        for i in range(n_batches):
            batch_xs, batch_ys = (
                train_set["X"][idx[
                    i * self.cfg.batch_sz:
                    (i + 1) * self.cfg.batch_sz]],
                train_set["y"][idx[
                    i * self.cfg.batch_sz:
                    (i + 1) * self.cfg.batch_sz]])
            self.train_on_batch(sess, writer, batch_xs, batch_ys)
        print("Evaluating on dev set")
        dev_metric = self.compute_dev_metric(
            sess, dev_set["X"], dev_set["y"])
        print("- dev acc: {:.2f}".format(dev_metric * 100.0))
        return dev_metric

    def fit(self, sess, writer, train_set, dev_set, saver=None):
        best_dev_acc = 0
        self.iter = 0
        for epoch in range(self.cfg.n_epochs):
            print("Epoch {:} out of {:}".format(
                epoch + 1, self.cfg.n_epochs))
            dev_acc = self.run_epoch(
                sess, writer, train_set, dev_set)
            if dev_acc > best_dev_acc:
                best_dev_acc = dev_acc
                if saver is not None:
                    print("New best dev acc! Saving model in "
                          "./model.weights")
                    saver.save(sess, './model.weights')

    def __init__(self, cfg):
        self.cfg = cfg
        self.build()
