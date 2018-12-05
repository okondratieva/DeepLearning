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
        'batch_size': 10,
        'learning_rate': 0.0001,
        'num_epoch': 22,
        'size': (432, 288)
    }

    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)
    for handler in logger.handlers[:]:
        handler.flush()
        logger.removeHandler(handler)
    logger.addHandler(logging.StreamHandler())
    logger.addHandler(logging.FileHandler('additional' + '.log', mode = 'w'))

    logger.info("batch_size: {}".format(config['batch_size']))
    logger.info("learning_rate: {}".format(config['learning_rate']))
    logger.info("num_epoch: {}".format(config['num_epoch']))

    #additional network
    dataset = dtstMainB(config['size'], config['batch_size'])

    convolution = ConvolutionPart()
    network = ClassificatorPart(input = convolution, name = 'classB')
    save_image(network, 'additional', (1,) + (3,) + config['size'])
    module = mx.mod.Module(symbol = network, context=mx.gpu())
    module.fit(
        dataset['train'],
        eval_data = dataset['test'],
        optimizer = 'sgd',
        optimizer_params = {'learning_rate':config['learning_rate']},
        initializer = mx.initializer.Xavier(),
        eval_metric = 'acc',
        batch_end_callback = mx.callback.Speedometer(config['batch_size'], 100),
        num_epoch = config['num_epoch']
    )

    parse_log('additional' + '.log', 'additional')


    arg_params, aux_params = module.get_params()
    save_dict = {('arg:%s' % k) : v.as_in_context(mx.cpu()) for k, v in arg_params.items()}
    save_dict.update({('aux:%s' % k) : v.as_in_context(mx.cpu()) for k, v in aux_params.items()})
    mx.ndarray.save('additional.params', save_dict)