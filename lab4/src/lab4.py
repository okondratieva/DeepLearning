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
    def __init__(self, iterator, size):
        self._iterator = iterator
        self._batch_size = iterator.provide_label[0][1][0]
        self._size = size
        self._labels = [mx.nd.zeros((self._batch_size, self._size))]

    def __iter__(self):
        return self

    def __next__(self):
        return self.next()

    def next(self):
        return mx.io.DataBatch(self._iterator.next().data, self._labels)

    def reset(self):
        self._iterator.reset()
    
    @property
    def provide_data(self):
        return self._iterator.provide_data

    @property
    def provide_label(self):
        return [mx.io.DataDesc('autoencoder_label', (self._batch_size, self._size), np.float32)]


def addLayer(network, num, dataset, arg_params = {}, name = None):
    logger = logging.getLogger()

    logger.info('add: {}'.format(num))

    output = mx.sym.FullyConnected(data = network, num_hidden = num, name = 'fc' + str(num))
    output = mx.sym.Activation(data = output, act_type = 'relu')

    decoder_size = network.infer_shape(data = (1,) + (3,) + config['size'])[1][0][1]
    autoencoder = mx.sym.FullyConnected(data = output, num_hidden = decoder_size, name = 'autoencoder_fc')
    autoencoder = mx.sym.Activation(data = autoencoder, act_type = 'relu', name = 'autoencoder_act')

    autoencoder = network - autoencoder
    autoencoder = mx.sym.square(autoencoder)

    labels = mx.sym.var('autoencoder_label')

    autoencoder = mx.sym.MAERegressionOutput(autoencoder, labels)

    if name is None:
        name = 'autoencoder' + str(num)

    save_image(autoencoder, name, (1,) + (3,) + config['size'])

    _dataset = AutoencoderIter(dataset, decoder_size)
    print(_dataset.provide_label)

    module = mx.mod.Module(symbol = autoencoder, context = mx.gpu(), label_names=['autoencoder_label'], fixed_param_names = arg_params.keys())
    logger.info("learn autoencoder: {}".format(num))
    module.fit(
        _dataset,
        optimizer = 'sgd',
        optimizer_params = {'learning_rate':config['learning_rate'] * 10},
        eval_metric='mae',
        arg_params=arg_params,
        aux_params={},
        allow_missing=True,
        initializer = mx.initializer.Xavier(),
        batch_end_callback = mx.callback.Speedometer(config['batch_size'], 100),
        num_epoch = config['num_epoch'] // 4)

    arg_params = module.get_params()[0]
    
    return output, dict({k:arg_params[k] for k in arg_params if 'autoencoder' not in k})

def pretrainFullyConnected(config, dataset):
    output = mx.sym.var('data')
    output = mx.sym.flatten(data = output, name = 'flatten')
    arg_params = {}

    for num in config:
        output, arg_params = addLayer(output, num, dataset, arg_params)

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

dataset = dtstMain(config['size'], 10)

network, arg_params = pretrainFullyConnected([2300, 1150, 575, 288, 144, 72, 36], dataset = dataset['train'])

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