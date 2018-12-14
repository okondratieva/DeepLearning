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
    'learning_rate': 0.004,
    'num_epoch': 120,
    'size': (224, 224)
}

logger = logging.getLogger()
logger.setLevel(logging.DEBUG)
for handler in logger.handlers[:]:
    handler.flush()
    logger.removeHandler(handler)
logger.addHandler(logging.StreamHandler())
logger.addHandler(logging.FileHandler('transfer1' + '.log', mode = 'w'))
logger.info("batch_size: {}".format(config['batch_size']))
logger.info("learning_rate: {}".format(config['learning_rate']))
logger.info("num_epoch: {}".format(config['num_epoch']))

#main network
dataset = dtstMain(config['size'], config['batch_size'])
# arg_params, aux_params = load_params()

network = symbol()
save_image(network, 'transfer1', (1,) + (3,) + config['size'])
module = mx.mod.Module(symbol = network, context=mx.gpu())#, fixed_param_names = arg_params.keys())

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

module.fit(
    dataset['train'],
    optimizer = 'sgd',
    optimizer_params = {'learning_rate':config['learning_rate']},
    eval_metric='acc',
    eval_data=dataset['test'],
    initializer = mx.initializer.Xavier(),
    batch_end_callback = mx.callback.Speedometer(config['batch_size'], 100),
    num_epoch = config['num_epoch']
)

parse_log('transfer1' + '.log', 'transfer1')