import mxnet as mx
import os
import sys
import logging

config = {
    'name': 'conv2',
    'dataset': 'main',
    'size': (432, 288),
    'batch_size': 10,
    'learning_rate': 0.0001,
    'num_epoch': 60
}

logger = logging.getLogger()
logger.setLevel(logging.DEBUG)
logger.addHandler(logging.FileHandler(config['name'] + '.log'))
logger.addHandler(logging.StreamHandler())

for key in config:
    logger.info('{}: {}'.format(key, config[key]))

file_path = os.path.dirname(__file__)
project_path = os.path.join(file_path, '..', '..')

sys.path.append(project_path)
from load_dataset import *

if config['dataset'] == 'mnist':
    dataset = dtstMNIST(config['size'], config['batch_size'])
elif config['dataset'] == 'example':
    dataset = dtstExample(config['size'], config['batch_size'])
elif config['dataset'] == 'main':
    dataset = dtstMain(config['size'], config['batch_size'])
else:
    logger.error('unrecognized dataset')
    exit()

def FCNN(input, configuration = [], default_activation = 'relu'):
    _hidden_layers = []
    for layer in configuration:
        if len(layer) == 1:
            _hidden_layers.append((layer[0], default_activation))
        else:
            _hidden_layers.append((layer[0], layer[1]))

    _input = mx.sym.flatten(data=input)

    fc = _input
    for layer in _hidden_layers:
        fc = mx.sym.FullyConnected(data = fc, num_hidden = layer[0])
        fc = mx.sym.Activation(data = fc, act_type = layer[1])

    output = mx.sym.FullyConnected(data = fc, num_hidden = dataset['classes_num'])
    output = mx.sym.SoftmaxOutput(data = output, name = 'softmax')

    return output

def block(input):
    output = mx.sym.Convolution(data = input, kernel = (5, 5), num_filter = 20, pad = (2, 2))
    output = mx.sym.Activation(data = output, act_type = 'relu')
    output = mx.sym.Pooling(data = output, pool_type = 'max', kernel = (2, 2), stride = (2, 2))
    return output

input = mx.sym.var('data')
network = block(input)
network = block(network)
network = block(network)
network = block(network)
network = mx.sym.Convolution(data = network, kernel = (5, 5), num_filter = 10, pad = (2, 2))
network = mx.sym.Activation(data = network, act_type = 'relu')
network = FCNN(network, [(486,), (243,), (121,), (120,), (60,), (30,)])

image = mx.viz.plot_network(network, save_format='png')
image.render(config['name'])

model = mx.mod.Module(symbol = network, context = mx.gpu())

model.fit(
    dataset['train'],
    eval_data = dataset['test'],
    optimizer = 'sgd',
    optimizer_params = {'learning_rate':config['learning_rate']},
    initializer = mx.initializer.Xavier(),
    eval_metric = 'acc',
    batch_end_callback = mx.callback.Speedometer(config['batch_size'], 100),
    num_epoch = config['num_epoch']
)

for key in config:
    logger.info('{}: {}'.format(key, config[key]))

# net.save_params(config['name'] + '.net')