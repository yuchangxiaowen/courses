## LSF + Pytorch实现单机版GPU加速

### Python代码的书写

1. 无论GPU与否，首先定义网络与数据

```python
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader


# Parameters and DataLoaders
input_size = 5
output_size = 2
batch_size = 30
data_size = 100


# Dataloader
class RandomDataset(Dataset):
    def __init__(self, size, length):
        self.len = length
        self.data = torch.randn(length, size)

    def __getitem__(self, index):
        return self.data[index]

    def __len__(self):
        return self.len

rand_loader = DataLoader(dataset=RandomDataset(input_size, 100),
                         batch_size=batch_size, shuffle=True)


# neural network
class Model(nn.Module):
    def __init__(self, input_size, output_size):
        super(Model, self).__init__()
        self.fc = nn.Linear(input_size, output_size)

    def forward(self, input):
        output = self.fc(input)
        print("\tIn Model: input size", input.size(),
              "output size", output.size())

        return output

model = Model(input_size, output_size)
```

2. 在CPU下，直接喂数据给模型进行计算；在单GPU下，将模型参数和数据放置到GPU下，进行计算；在多GPU下，首先将模型参数放置到0号GPU下，然后利用`DataParallel`函数将参数的副本拷贝到其他GPU下，最后每轮训练时通过0号GPU将数据广播至其他GPU。将以上三种情况融合即为以下代码（简化起见，以下代码仅展示了前向传播部分）

```python
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
if torch.cuda.device_count() > 1:
    print("Let's use", torch.cuda.device_count(), "GPUs!")
    # dim = 0 [30, xxx] -> [10, ...], [10, ...], [10, ...] on 3 GPUs
    model = nn.DataParallel(model)
model.to(device)

for data in rand_loader:
    input = data.to(device)
    output = model(input)
    print("Outside: input size", input.size(),
          "output_size", output.size())

```

### lsf代码的书写

新建`filename.sh`，通过以下lsf代码实现GPU加速。为方便起见，假设文件名为`test_torch.sh`

```shell
#/bin/bash
#BSUB -J test_torch
#BSUB -e %J.err
#BSUB -o %J.out
#BSUB -n 1
#BSUB -q gauss
#BSUB -R "select [ngpus>0] rusage [ngpus_excl_p=4]"
python /your/path/your_python_file.py
```

说明：
1. 第二行为任务名称，此处随意
2. 第三行对应报错输出的文件，第四行对应标准输出的文件。以上代码表示用任务编号命名相应输出文件，如`300.err`、`300.out`
3. 修改倒数第二行`ngpus_excl_p=`后的数字，选择1-4个GPU进行计算
4. 修改最后一行，填入要运行的python文件的路径
5. 在命令行内执行`bsub < test_torch.sh`。通过`bjobs`命令查看任务执行状态，在`.err`与`.out`文件中查看相应的输出

运行结果如下
```
Let's use 4 GPUs!
        In Model: input size torch.Size([6, 5]) output size torch.Size([6, 2])
        In Model: input size torch.Size([8, 5]) output size torch.Size([8, 2])
        In Model: input size torch.Size([8, 5]) output size torch.Size([8, 2])
        In Model: input size torch.Size([8, 5]) output size torch.Size([8, 2])
Outside: input size torch.Size([30, 5]) output_size torch.Size([30, 2])
        In Model: input size torch.Size([8, 5]) output size torch.Size([8, 2])
        In Model: input size torch.Size([8, 5]) output size torch.Size([8, 2])
        In Model: input size torch.Size([8, 5]) output size torch.Size([8, 2])
        In Model: input size torch.Size([6, 5]) output size torch.Size([6, 2])
Outside: input size torch.Size([30, 5]) output_size torch.Size([30, 2])
        In Model: input size torch.Size([8, 5]) output size torch.Size([8, 2])
        In Model: input size torch.Size([8, 5]) output size torch.Size([8, 2])
        In Model: input size torch.Size([8, 5]) output size torch.Size([8, 2])
        In Model: input size torch.Size([6, 5]) output size torch.Size([6, 2])
Outside: input size torch.Size([30, 5]) output_size torch.Size([30, 2])
        In Model: input size torch.Size([3, 5]) output size torch.Size([3, 2])
        In Model: input size torch.Size([3, 5]) output size torch.Size([3, 2])
        In Model: input size torch.Size([3, 5]) output size torch.Size([3, 2])
        In Model: input size torch.Size([1, 5]) output size torch.Size([1, 2])
Outside: input size torch.Size([10, 5]) output_size torch.Size([10, 2])
```

## LSF + Tensorflow实现单机版GPU加速

### Python代码的书写

```python
import numpy as np
from tensorflow.python.client import device_lib
import tensorflow as tf
import time


def fetch_data(n, test):
    train_x = []
    train_y = []
    test_x = []
    test_y = []
    n_test = int(n * test)
    n_train = n - n_test

    for i in range(n_train):
        theta = np.random.random() * 2 * np.pi
        tmp = np.random.random() * 2
        r = np.sqrt(tmp if tmp <= 1 else tmp + 1)
        train_x.append([r * np.sin(theta), r * np.cos(theta)])
        train_y.append(1 if r <= 1 else 0)

    for i in range(n_test):
        theta = np.random.random() * 2 * np.pi
        tmp = np.random.random() * 2
        r = np.sqrt(tmp if tmp <= 1 else tmp + 1)
        test_x.append([r * np.sin(theta), r * np.cos(theta)])
        test_y.append(1 if r <= 1 else 0)

    train_x = np.array(train_x)
    train_y = np.array(train_y)
    test_x = np.array(test_x)
    test_y = np.array(test_y)

    tmp = train_y.copy()
    train_y = np.zeros((tmp.shape[0], 2))
    train_y[np.arange(tmp.shape[0]), tmp] = 1

    tmp = test_y.copy()
    test_y = np.zeros((tmp.shape[0], 2))
    test_y[np.arange(tmp.shape[0]), tmp] = 1

    return train_x, train_y, test_x, test_y


def tower_metrics(scope, x_batch, y_batch):
    hidden_sz_1 = 4
    hidden_sz_2 = 2

    W_0 = tf.get_variable(
        'W_0', [2, hidden_sz_1],
        initializer=tf.orthogonal_initializer())
    b_0 = tf.get_variable('b_0', [hidden_sz_1])
    W_1 = tf.get_variable(
        'W_1', [hidden_sz_1, hidden_sz_2],
        initializer=tf.orthogonal_initializer())
    b_1 = tf.get_variable('b_1', [hidden_sz_2])
    W_2 = tf.get_variable(
        'W_2', [hidden_sz_2, 2],
        initializer=tf.orthogonal_initializer())
    b_2 = tf.get_variable('b_2', [2])
    y = tf.nn.softmax(tf.matmul(
        tf.tanh(tf.matmul(
            tf.tanh(tf.matmul(x_batch, W_0) + b_0),
            W_1) + b_1),
        W_2) + b_2)
    cross_entropy = tf.reduce_mean(
        -tf.reduce_sum(y_batch * tf.log(y), reduction_indices=[1]))

    correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_batch, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    return cross_entropy, accuracy


def average_gradients(tower_grads):
    """Calculate the average gradient for each shared variable across all towers.

    Note that this function provides a synchronization point across all towers.
    
    Args:
        tower_grads: List of lists of (gradient, variable) tuples. The outer list
            is over individual gradients. The inner list is over the gradient
            calculation for each tower.
    Returns:
        List of pairs of (gradient, variable) where the gradient has been averaged
        across all towers.
    """
    average_grads = []
    for grad_and_vars in zip(*tower_grads):
        # Note that each grad_and_vars looks like the following:
        #   ((grad0_gpu0, var0_gpu0), ... , (grad0_gpuN, var0_gpuN))
        grads = []
        for g, _ in grad_and_vars:
            # Add 0 dimension to the gradients to represent the tower.
            expanded_g = tf.expand_dims(g, 0)

            # Append on a 'tower' dimension which we will average over below.
            grads.append(expanded_g)

        # Average over the 'tower' dimension.
        grad = tf.concat(axis=0, values=grads)
        grad = tf.reduce_mean(grad, 0)

        # Keep in mind that the Variables are redundant because they are shared
        # across towers. So .. we will just return the first tower's pointer to
        # the Variable.
        v = grad_and_vars[0][1]
        grad_and_var = (grad, v)
        average_grads.append(grad_and_var)
    return average_grads


def average_metric(metrics):
    metric = tf.concat(axis=0, values=[tf.expand_dims(metric, 0) for metric in metrics])
    return tf.reduce_mean(metric, 0)


def build_model(train_x, train_y, test_x, test_y):
    gpu_list = get_available_gpus()

    # prepare input queue
    with tf.device('/cpu:0'):
        lr_rate = .03
        optimizer = tf.train.AdamOptimizer(learning_rate=lr_rate)

        # train data
        batch_sz = 10
        train_queue = tf.train.slice_input_producer(
            [tf.cast(train_x, tf.float32), tf.cast(train_y, tf.float32)],
            shuffle=False)
        x_batch, y_batch = tf.train.batch(
            train_queue, batch_size=batch_sz,
            num_threads=4, capacity=2 * len(gpu_list))
        batch_queue = tf.contrib.slim.prefetch_queue.prefetch_queue(
              [x_batch, y_batch], capacity=2 * len(gpu_list))

        # test data
        test_sz = test_x.shape[0] // len(gpu_list)
        test_xs = [tf.cast(test_x[i * test_sz: (i + 1) * test_sz], tf.float32) for i in range(len(gpu_list))]
        test_ys = [tf.cast(test_y[i * test_sz: (i + 1) * test_sz], tf.float32) for i in range(len(gpu_list))]

        # compute gradient for each gpu
        tower_grads = []
        tower_losses = []
        tower_accs = []
        tower_accs_test = []
        with tf.variable_scope(tf.get_variable_scope()):
            for idx, gpu_device in enumerate(gpu_list):
                with tf.device(gpu_device):
                    with tf.name_scope('tower_{}'.format(idx)) as scope:
                        x_batch, y_batch = batch_queue.dequeue()
                        loss, acc = tower_metrics(scope, x_batch, y_batch)
                        tf.get_variable_scope().reuse_variables()
                        tower_grads.append(optimizer.compute_gradients(loss))
                        tower_losses.append(loss)
                        tower_accs.append(acc)

                    with tf.name_scope('tower_test_{}'.format(idx)) as scope:
                        _, acc = tower_metrics(scope, test_xs[idx], test_ys[idx])
                        tf.get_variable_scope().reuse_variables()
                        tower_accs_test.append(acc)


        # We must calculate the mean of each gradient. Note that this is the
        # synchronization point across all towers.
        gvs = average_gradients(tower_grads)
        loss = average_metric(tower_losses)
        acc = average_metric(tower_accs)
        acc_test = average_metric(tower_accs_test)

        # Apply updates to Variables
        capped_gvs = [
            (None if grad is None else tf.clip_by_value(grad, -1., 1.), var)
            for grad, var in gvs]
        train_op = optimizer.apply_gradients(capped_gvs)
        
        summary_op = tf.summary.merge([
            tf.summary.scalar("loss", loss),
            tf.summary.scalar("accuracy", acc)
        ])

    return train_op, summary_op, acc_test


def get_available_gpus():
    local_device_protos = device_lib.list_local_devices()
    return [x.name for x in local_device_protos if x.device_type == 'GPU']


def main():
    train_x, train_y, test_x, test_y = fetch_data(n=400, test=.5)
    train_op, summary_op, test_acc = build_model(train_x, train_y, test_x, test_y)

    n_epoch = 300
    init = tf.global_variables_initializer()

    with tf.Session(config=tf.ConfigProto(allow_soft_placement=True)) as sess:
        sess.run(init)

        # Start the queue runners.
        tf.train.start_queue_runners(sess=sess)

        writer = tf.summary.FileWriter(
            "/nfsshare/home/caoshengyu/workplace/tmp/{}".format(
                time.strftime("%Y-%m-%d-%H-%M-%S", time.localtime())
        ))
        writer.add_graph(sess.graph)

        for i in range(n_epoch):
            s, _ = sess.run([summary_op, train_op])
            writer.add_summary(s, i)

        print("Testing accuracy is {}.".format(sess.run(test_acc)))


if __name__ == '__main__':
    main()

```

### lsf代码的书写

新建`filename.sh`，通过以下lsf代码实现GPU加速。为方便起见，假设文件名为`test_tf.sh`

```shell
#/bin/bash
#BSUB -J test_tf
#BSUB -e %J.err
#BSUB -o %J.out
#BSUB -n 1
#BSUB -q gauss
#BSUB -R "select [ngpus>0] rusage [ngpus_excl_p=4]"
python /your/path/your_python_file.py
```

说明：
1. 第二行为任务名称，此处随意
2. 第三行对应报错输出的文件，第四行对应标准输出的文件。以上代码表示用任务编号命名相应输出文件，如`300.err`、`300.out`
3. 修改倒数第二行`ngpus_excl_p=`后的数字，选择1-4个GPU进行计算
4. 修改最后一行，填入要运行的python文件的路径
5. 在命令行内执行`bsub < test_tf.sh`。通过`bjobs`命令查看任务执行状态，在`.err`与`.out`文件中查看相应的输出
