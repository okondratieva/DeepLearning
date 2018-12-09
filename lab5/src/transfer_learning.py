import mxnet as mx
import os
import sys

project_path = os.path.dirname(__file__)
project_path = os.path.join(project_path, '..', '..')
sys.path.append(project_path)
from blocks import *
from fit import *
from parse_log import *

def ConvolutionPart():
    output = Convolution(64, conv_size=3)
    output = Convolution(64, conv_size = 3, input=output)
    output = Convolution(128, conv_size = 3, input=output)
    output = Inseption(input = output, pooling = False)
    output = Inseption(input = output, coeff = 2)
    output = mx.sym.Convolution(data = output, kernel = (3, 3), stride = (1, 1), pad = (1, 1), num_filter = 128)
    output = mx.sym.Activation(data = output, act_type = 'relu')
    return output

def ClassificatorPart(input = None, name = 'classA'):
    if input is None:
        input = mx.sym.var('data')
    output = mx.sym.flatten(data=input)

    output = mx.sym.FullyConnected(data = output, num_hidden = 6220, name = name + '_fc_' + str(6220))
    output = mx.sym.Activation(data = output, act_type = 'relu', name = name + '_act_' + str(6220))

    output = mx.sym.FullyConnected(data = output, num_hidden = 622, name = name + '_fc_' + str(622))
    output = mx.sym.Activation(data = output, act_type = 'relu', name = name + '_act_' + str(622))

    output = mx.sym.FullyConnected(data = output, num_hidden = 14, name = name + '_fc_' + str(14))
    output = mx.sym.SoftmaxOutput(data = output, name = 'softmax')

    return output

def FixedParams():
    params = []
    for i in range(16):
        params.append('convolution' + str(i) + '_weight')
        params.append('convolution' + str(i) + '_bias')
    return params

if __name__ == "__main__":
    config = {
        'name': 'vgg16',
        'batch_size': 10,
        'learning_rate': 0.0001,
        'num_epoch': 22,
        'size': (432, 288)
    }

    network = mx.sym.load(config['name'] + '.json')

    save_image(network, config['name'], (1, 3, 244, 244))