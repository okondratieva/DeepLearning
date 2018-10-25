import mxnet as mx
import os
import sys
import logging

config = {
    'name': 'name',
    'dataset': 'mnist',
    'size': (256, 256),
    'batch_size': 1,
    'learning_rate': 0.01,
    'num_epoch': 10,
    'net_config': [],
    'default_activation': 'tanh'
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

def FCNN():
    _hidden_layers = []
    for layer in config['net_config']:
        if len(layer) == 1:
            _hidden_layers.append((layer[0], config['default_activation']))
        else:
            _hidden_layers.append((layer[0], layer[1]))

    input = mx.sym.var('data')
    input = mx.sym.flatten(data=input)

    fc = input
    for layer in _hidden_layers:
        fc = mx.sym.FullyConnected(data = fc, num_hidden = layer[0])
        fc = mx.sym.Activation(data = fc, act_type = layer[1])

    output = mx.sym.FullyConnected(data = fc, num_hidden = dataset['classes_num'])
    output = mx.sym.SoftmaxOutput(data = output, name = 'softmax')

    return output


net = mx.mod.Module(symbol = FCNN(), context = mx.gpu())

net.fit(
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

net.save_params(config['name'] + '.net')