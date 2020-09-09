import tensorflow as tf
import numpy as np
from scipy.io import loadmat
import os

from config import Config
from model import Model


def load_data(cfg):
    data_file = loadmat(
        os.path.join(cfg.root_dir, "data.mat"),
        squeeze_me=True,
        struct_as_record=False)
    data = data_file["data"]

    train_set = {
        "X": np.float32(data.training.inputs).T,
        "y": np.float32(data.training.targets).T
    }
    dev_set = {
        "X": np.float32(data.validation.inputs).T,
        "y": np.float32(data.validation.targets).T
    }
    test_set = {
        "X": np.float32(data.test.inputs).T,
        "y": np.float32(data.test.targets).T
    }
    return train_set, dev_set, test_set


def main():
    cfg = Config()
    train_set, dev_set, test_set = load_data(cfg)

    with tf.Graph().as_default() as graph:
        with tf.device('/gpu:0'):
            with tf.name_scope("model"):
                model = Model(cfg)
            init_op = tf.global_variables_initializer()
            saver = tf.train.Saver()
    graph.finalize()

    writer = tf.summary.FileWriter(os.path.join(cfg.root_dir, "tmp"))
    sess_cfg=tf.ConfigProto(allow_soft_placement=True)
    with tf.Session(graph=graph, config=sess_cfg) as session:
        writer.add_graph(session.graph)
        session.run(init_op)

        model.fit(session, writer, train_set, dev_set, saver=saver)

        saver.restore(session, os.path.join(cfg.root_dir, "model.weights"))
        test_metric = model.compute_dev_metric(
            session, test_set["X"], test_set["y"])
        print("Final test acc: {:.2f}".format(test_metric * 100.0))


if __name__ == "__main__":
    main()
