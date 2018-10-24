import mxnet as mx
import os
import sys

name = 'test4'

class Logger(object):
    def __init__(self):
        self.terminal = sys.stdout
        self.log = open(name + '.log', "a")

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)

    def flush(self):
        pass 

sys.stdout = Logger()

file_path = os.path.dirname(__file__)
dataset_size = 'reduce' #'origin'
dataset_calsses_num = 14 #86
learning_rate = 0.01
print('learning_rate: ', learning_rate)
epoch_num = 10
print('epoch_num: ', epoch_num)
pack_root = os.path.join(file_path, '..', '..', 'dataset', 'pack', dataset_size)
shape = (3, 432, 288)

batch_size = 1
print('batch_size: ', batch_size)

def dataset(train, test, classes_num):
    return {'train':train, 'test':test, 'classes_num':classes_num}

def dtstMNIST():
    mnist = mx.test_utils.get_mnist()
    print('dataset: ', 'MNIST')
    return dataset(
        mx.io.NDArrayIter(mnist['train_data'], mnist['train_label'], batch_size, shuffle=True),
        mx.io.NDArrayIter(mnist['test_data'], mnist['test_label'], batch_size),
        10
    )

def dtstExample():
    train = mx.io.ImageRecordIter(
        path_imgrec = os.path.join(pack_root, 'example_train.rec'),
        path_imgidx = os.path.join(pack_root, 'example_train.idx'),
        data_shape = shape,
        batch_size = batch_size,
        shuffle = True
    )
    test = mx.io.ImageRecordIter(
        path_imgrec = os.path.join(pack_root, 'example_val.rec'),
        path_imgidx = os.path.join(pack_root, 'example_val.idx'),
        data_shape = shape,
        batch_size = batch_size
    )
    print('dataset: ', dataset_size, ' example')
    return dataset(train, test, dataset_calsses_num)

def dtstMain():
    train = mx.io.ImageRecordIter(
        path_imgrec = os.path.join(pack_root, 'train.rec'),
        path_imgidx = os.path.join(pack_root, 'train.idx'),
        data_shape = shape,
        batch_size = batch_size,
        shuffle = True
    )
    test = mx.io.ImageRecordIter(
        path_imgrec = os.path.join(pack_root, 'test.rec'),
        path_imgidx = os.path.join(pack_root, 'test.idx'),
        data_shape = shape,
        batch_size = batch_size
    )
    print('dataset: ', dataset_size, ' main')
    return dataset(train, test, dataset_calsses_num)

def FCNN(hidden_layers, dataset, default_activation='tanh'):
    _hidden_layers = []
    for layer in hidden_layers:
        if len(layer) == 1:
            _hidden_layers.append((layer[0], default_activation))
        else:
            _hidden_layers.append((layer[0], layer[1]))
    print('hidden layers conf: ',_hidden_layers)

    input = mx.sym.var('data')
    input = mx.sym.flatten(data=input)

    fc = input
    for layer in _hidden_layers:
        fc = mx.sym.FullyConnected(data = fc, num_hidden = layer[0])
        fc = mx.sym.Activation(data = fc, act_type = layer[1])

    output = mx.sym.FullyConnected(data = fc, num_hidden = dataset['classes_num'])
    output = mx.sym.SoftmaxOutput(data = output, name = 'softmax')

    return output

dataset = dtstMNIST()
net_shema = FCNN([
        (80,),
    ],
    dataset)
net = mx.mod.Module(symbol = net_shema, context = mx.gpu())

import logging
logger = logging.getLogger()
logger.setLevel(logging.DEBUG)
logger.addHandler(logging.FileHandler(name + '.log'))
logger.addHandler(logging.StreamHandler())

net.fit(
    dataset['train'],
    eval_data = dataset['test'],
    optimizer = 'sgd',
    optimizer_params = {'learning_rate':learning_rate},
    initializer = mx.initializer.Xavier(),
    eval_metric = 'acc',
    batch_end_callback = mx.callback.Speedometer(batch_size, 100),
    num_epoch = epoch_num
)

net.save_params(name + '.net')