import mxnet as mx
import os
import sys

project_path = os.path.dirname(__file__)
project_path = os.path.join(project_path, '..', '..')
sys.path.append(project_path)
from blocks import *
from fit import *
from transfer_learning import *
from parse_log import *

config = {
    'batch_size': 10,
    'learning_rate': 0.0001,
    'num_epoch': 60,
    'size': (432, 288)
}

# logger = logging.getLogger()
# logger.setLevel(logging.DEBUG)
# for handler in logger.handlers[:]:
#     handler.flush()
#     logger.removeHandler(handler)
# logger.addHandler(logging.StreamHandler())
# logger.addHandler(logging.FileHandler('transfer3' + '.log', mode = 'w'))
# logger.info("batch_size: {}".format(config['batch_size']))
# logger.info("learning_rate: {}".format(config['learning_rate']))
# logger.info("num_epoch: {}".format(config['num_epoch']))

# #main network
# dataset = dtstMain(config['size'], config['batch_size'])

# network = ConvolutionPart()
# network = ClassificatorPart(input = network, name = 'classB')
# save_image(network, 'transfer3', (1,) + (3,) + config['size'])
# module = mx.mod.Module(symbol = network, context=mx.gpu())

# save_dict = mx.ndarray.load('additional.params')
# arg_params = {}
# aux_params = {}
# for k, value in save_dict.items():
#     arg_type, name = k.split(':', 1)
#     if arg_type == 'arg':
#         arg_params[name] = value
#     elif arg_type == 'aux':
#         aux_params[name] = value
#     else:
#         raise ValueError("Invalid param file " + 'additional.params')

# module.fit(
#     dataset['train'],
#     optimizer = 'sgd',
#     optimizer_params = {'learning_rate':config['learning_rate']},
#     eval_metric='acc',
#     eval_data=dataset['test'],
#     arg_params=arg_params,
#     aux_params=aux_params,
#     allow_missing=True,
#     initializer = mx.initializer.Xavier(),
#     batch_end_callback = mx.callback.Speedometer(config['batch_size'], 100),
#     num_epoch = config['num_epoch']
# )

parse_log('transfer3' + '.log', 'transfer3')