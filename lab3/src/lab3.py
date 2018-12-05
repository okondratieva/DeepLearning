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

def cnn0():
    output = Convolution(3, conv_size=3)
    output = FullyConnected([(3110,), (311,), (31,)], input = output)
    return output

def cnn1():
    output = Convolution(3, conv_size=3)
    output = Convolution(3, conv_size = 3, input=output)
    output = Convolution(3, conv_size = 3, input=output)
    output = mx.sym.Convolution(data = output, kernel = (3, 3), stride = (1, 1), pad = (1, 1), num_filter = 3)
    output = mx.sym.Activation(data = output, act_type = 'relu')
    output = FullyConnected([(583,), (58,)], input = output)
    return output

def cnn2():
    output = Convolution(10, conv_size=3)
    output = Convolution(10, conv_size = 3, input=output)
    output = Convolution(10, conv_size = 3, input=output)
    output = mx.sym.Convolution(data = output, kernel = (3, 3), stride = (1, 1), pad = (1, 1), num_filter = 10)
    output = mx.sym.Activation(data = output, act_type = 'relu')
    output = FullyConnected([(1944,), (194,)], input = output)
    return output

def cnn3():
    output = Convolution(40, conv_size=3)
    output = Convolution(30, conv_size = 3, input=output)
    output = Convolution(20, conv_size = 3, input=output)
    output = mx.sym.Convolution(data = output, kernel = (3, 3), stride = (1, 1), pad = (1, 1), num_filter = 10)
    output = mx.sym.Activation(data = output, act_type = 'relu')
    output = FullyConnected([(1944,), (194,)], input = output)
    return output

def cnn4():
    input = mx.sym.var('data')
    output = Convolution(40, conv_size=3, input=input)
    output = Convolution(30, conv_size = 3, input=output)
    output = Convolution(20, conv_size = 3, input=output)
    output = mx.sym.Convolution(data = output, kernel = (3, 3), stride = (1, 1), pad = (1, 1), num_filter = 10)
    output = mx.sym.Activation(data = output, act_type = 'relu')

    resize = mx.sym.Pooling(data = input, kernel = (2, 2), stride = (2, 2), pool_type = 'avg')
    resize = mx.sym.Pooling(data = resize, kernel = (2, 2), stride = (2, 2), pool_type = 'avg')
    resize = mx.sym.Pooling(data = resize, kernel = (2, 2), stride = (2, 2), pool_type = 'avg')

    output = mx.sym.concat(output, resize)

    output = FullyConnected([(2527,), (252,)], input = output)
    return output

def cnn5():
    output = SmallInseption()
    output = Convolution(10, conv_size=3, input=output)
    output = Convolution(10, conv_size=3, input=output)

    output = FullyConnected([(1944,), (194,)], input = output)
    return output

def cnn6():
    output = Convolution(128, conv_size=3)
    output = Convolution(64, conv_size = 3, input=output)
    output = Convolution(32, conv_size = 3, input=output)
    output = mx.sym.Convolution(data = output, kernel = (3, 3), stride = (1, 1), pad = (1, 1), num_filter = 10)
    output = mx.sym.Activation(data = output, act_type = 'relu')
    output = FullyConnected([(1944,), (194,)], input = output)
    return output

def cnn7():
    output = Convolution(64, conv_size=3)
    output = Convolution(64, conv_size = 3, input=output)
    output = Convolution(128, conv_size = 3, input=output)
    output = Inseption(input = output)
    output = mx.sym.Convolution(data = output, kernel = (3, 3), stride = (1, 1), pad = (1, 1), num_filter = 64)
    output = mx.sym.Activation(data = output, act_type = 'relu')
    output = FullyConnected([(3110,), (311,)], input = output)
    return output

def cnn8():
    output = Convolution(64, conv_size=3)
    output = Convolution(64, conv_size = 3, input=output)
    output = Convolution(128, conv_size = 3, input=output)
    output = Inseption(input = output, pooling = False)
    output = Inseption(input = output, coeff = 2)
    output = mx.sym.Convolution(data = output, kernel = (3, 3), stride = (1, 1), pad = (1, 1), num_filter = 64)
    output = mx.sym.Activation(data = output, act_type = 'relu')
    output = FullyConnected([(3110,), (311,)], input = output)
    return output

def cnn9():
    output = Convolution(64, conv_size=3)
    output = Convolution(64, conv_size = 3, input=output)
    output = Convolution(128, conv_size = 3, input=output)
    output = Inseption(input = output, pooling = False)
    output = Inseption(input = output, coeff = 2)
    output = mx.sym.Convolution(data = output, kernel = (3, 3), stride = (1, 1), pad = (1, 1), num_filter = 128)
    output = mx.sym.Activation(data = output, act_type = 'relu')
    output = FullyConnected([(6220,), (622,)], input = output)
    return output

def cnn10():
    input = mx.sym.var('data')
    output = Convolution(64, conv_size=3, input = input)
    input = mx.sym.Pooling(data = input, kernel = (2, 2), stride = (2, 2), pool_type = 'avg')
    output = mx.sym.concat(input, output)
    output = Convolution(64, conv_size = 3, input=output)
    input = mx.sym.Pooling(data = input, kernel = (2, 2), stride = (2, 2), pool_type = 'avg')
    output = mx.sym.concat(input, output)
    output = Convolution(128, conv_size = 3, input=output)
    input = mx.sym.Pooling(data = input, kernel = (2, 2), stride = (2, 2), pool_type = 'avg')
    output = mx.sym.concat(input, output)
    output = Inseption(input = output, pooling = False)
    output = Inseption(input = output, coeff = 2)
    input = mx.sym.Pooling(data = input, kernel = (2, 2), stride = (2, 2), pool_type = 'avg')
    output = mx.sym.concat(input, output)
    output = mx.sym.Convolution(data = output, kernel = (3, 3), stride = (1, 1), pad = (1, 1), num_filter = 128)
    output = mx.sym.Activation(data = output, act_type = 'relu')
    output = mx.sym.concat(input, output)
    output = FullyConnected([(6366,), (636,)], input = output)
    return output

def cnn11():
    output = Convolution(64, conv_size=3)
    output = Convolution(128, conv_size = 3, input=output)
    output = Inseption(input = output)
    output = Inseption(input = output, coeff = 2, pooling = False)
    output = Inseption(input = output, coeff = 2)
    output = mx.sym.Convolution(data = output, kernel = (3, 3), stride = (1, 1), pad = (1, 1), num_filter = 128)
    output = mx.sym.Activation(data = output, act_type = 'relu')
    output = FullyConnected([(6220,), (622,)], input = output)
    return output

networks = [
    cnn0(),
    cnn1(),
    cnn2(),
    cnn3(),
    cnn4(),
    cnn5(),
    cnn6(),
    cnn7(),
    cnn8(),
    cnn9(),
    cnn10(),
    cnn11()
]

counter = 0
for network in networks[counter:counter + 1]:
    fit(name = 'cnn' + str(counter), **config, network = network)
    counter += 1