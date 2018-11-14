import mxnet as mx
import os
import sys

project_path = os.path.dirname(__file__)
project_path = os.path.join(project_path, '..', '..')
sys.path.append(project_path)
from fit import *
from blocks import *

config = {
    'dataset': 'main',
    'batch_size': 10,
    'learning_rate': 0.0001,
    'num_epoch': 60,
    'size': (432, 288)
}

def cnn1():
    output = Convolution(3, conv_size=3)
    output = Convolution(3, conv_size = 3, input=output)
    output = Convolution(3, conv_size = 3, input=output)
    output = mx.sym.Convolution(data = output, kernel = (3, 3), stride = (1, 1), pad = (1, 1), num_filter = 3)
    output = mx.sym.Activation(data = output, act_type = 'relu')
    output = FullyConnected([(583,), (58,)], input = output)
    return output

networks = [
    cnn1()
]

counter = 1
for network in networks:
    fit(name = 'cnn' + str(counter), **config, network = network)
    counter += 1