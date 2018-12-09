import mxnet as mx
import numpy as np
import os
import sys

project_path = os.path.dirname(__file__)
project_path = os.path.join(project_path, '..', '..')
sys.path.append(project_path)
from fit import *
from blocks import *
from load_dataset import *

config = {
    'dataset': 'main',
    'batch_size': 10,
    'learning_rate': 0.0001,
    'num_epoch': 60,
    'size': (108, 72)
}

class AutoencoderIter:
    def __init__(self, dataset, symbol, arg_params):
        self._dataset = dataset
        self._symbol = symbol
        self._module = mx.mod.Module(self._symbol, context=mx.gpu())
        self._module.bind(self._dataset.provide_data)
        self._module.init_params(arg_params=arg_params, aux_params={}, allow_missing=True, force_init=True)

    def __iter__(self):
        return self

    def __next__(self):
        return self.next()

    def next(self):
        self._module.forward(self._dataset.next())
        output = self._module.get_outputs()
        return mx.io.DataBatch(output, output)

    def reset(self):
        self._dataset.reset()
    
    @property
    def provide_data(self):
        return [mx.io.DataDesc('data', self._symbol.infer_shape(data=dataset['train'].provide_data[0][1])[1][0], np.float32)]

    @property
    def provide_label(self):
        return [mx.io.DataDesc('autoencoder_label', self._symbol.infer_shape(data=dataset['train'].provide_data[0][1])[1][0], np.float32)]


def addLayer(data, network, num, dataset, arg_params = {}, name = None):
    logger = logging.getLogger()

    logger.info('add: {}'.format(num))

    output = mx.sym.FullyConnected(data = data, num_hidden = num, name = 'fc' + str(num))
    output = mx.sym.Activation(data = output, act_type = 'relu')

    decoder_size = network.infer_shape(data = (1,) + (3,) + config['size'])[1][0][1]
    autoencoder = mx.sym.FullyConnected(data = output, num_hidden = decoder_size, name = 'autoencoder_fc')
    autoencoder = mx.sym.Activation(data = autoencoder, act_type = 'relu', name = 'autoencoder_act')

    labels = mx.sym.var('autoencoder_label')

    autoencoder = mx.sym.LinearRegressionOutput(autoencoder, labels)

    if name is None:
        name = 'autoencoder' + str(num)

    _dataset = AutoencoderIter(dataset, network, arg_params)
    if num == 2300:
        save_image(autoencoder, name, dataset.provide_data[0][1])
    else:
        save_image(autoencoder, name, _dataset.provide_data[0][1])

    module = mx.mod.Module(symbol = autoencoder, context = mx.gpu(), label_names=['autoencoder_label'])
    logger.info("learn autoencoder: {}".format(num))
    module.fit(
        _dataset,
        optimizer = 'sgd',
        optimizer_params = {'learning_rate':config['learning_rate'] * 10},
        eval_metric='mse',
        arg_params=arg_params,
        aux_params={},
        allow_missing=True,
        initializer = mx.initializer.Xavier(),
        batch_end_callback = mx.callback.Speedometer(config['batch_size'], 100),
        num_epoch = config['num_epoch'] // 4)

    arg_params = dict(arg_params, **{'final_' + k:module.get_params()[0][k] for k in module.get_params()[0] if 'autoencoder' not in k})
    
    output = mx.sym.FullyConnected(data = network, num_hidden = num, name = 'final_fc' + str(num))
    output = mx.sym.Activation(data = output, act_type = 'relu')
    return output, arg_params

def pretrainFullyConnected(config, dataset):
    data = mx.sym.var('data')
    data = mx.sym.flatten(data = data, name = 'flatten')
    output = data
    arg_params = {}

    for num in config:
        output, arg_params = addLayer(data, output, num, dataset, arg_params)

    output = mx.sym.FullyConnected(data = output, num_hidden = 14)
    output = mx.sym.SoftmaxOutput(data = output, name = 'softmax')

    return output, arg_params

logger = logging.getLogger()
logger.setLevel(logging.DEBUG)
for handler in logger.handlers[:]:
    handler.flush()
    logger.removeHandler(handler)
logger.addHandler(logging.StreamHandler())
logger.addHandler(logging.FileHandler('pretrained' + '.log', mode = 'w'))

dataset = dtstMain(config['size'], config['batch_size'])

network, arg_params = pretrainFullyConnected([2300, 1150, 575, 288, 144, 72, 36], dataset = dataset['train'])
for name in arg_params:
    print(name)

save_image(network, 'pretrained', (1,) + (3,) + config['size'])

logger.info('train main network')

module = mx.mod.Module(symbol = network, context = mx.gpu())
module.fit(
        dataset['train'],
        eval_data=dataset['test'],
        eval_metric='acc',
        optimizer = 'sgd',
        optimizer_params = {'learning_rate':config['learning_rate']},
        arg_params=arg_params,
        aux_params={},
        allow_missing=True,
        initializer = mx.initializer.Xavier(),
        batch_end_callback = mx.callback.Speedometer(config['batch_size'], 100),
        num_epoch = config['num_epoch'])