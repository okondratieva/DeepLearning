import argparse
import os
from load_dataset import *
from parse_log import *
import logging
import mxnet as mx
import sys


def fit(network, *, name, dataset, batch_size, learning_rate, num_epoch, size = (210, 140), train = True, save_log = True, save_image = True, save_network = False):
    if len(sys.argv) > 1:
        save_log = True
        save_image = True
        train = True
        save_network = False

    parser = argparse.ArgumentParser()
    parser.add_argument('-n', '--name', type = str, default = name, help = 'name of experiment')
    parser.add_argument('-d', '--dataset', type = str, default = dataset, choices = ['mnist', 'example', 'main'], help = 'dataset to train')
    parser.add_argument('-s', '--size', type = int, nargs = 2, default = size, help = 'size of dataset (if \'main\' or \'example\' are choosen)')
    parser.add_argument('-bs', '--batch-size', type = int, default = batch_size, help = 'batch size')
    parser.add_argument('-lr', '--learning-rate', type = float, default = learning_rate, help = 'learning rate')
    parser.add_argument('-ne', '--num-epoch', type = int, default = num_epoch, help = 'number of epoch')
    parser.add_argument('--no-log', action = 'store_true', help = 'do not save the log file')
    parser.add_argument('--no-image', action = 'store_true', help = 'do not save the network\'s architecture in image')
    parser.add_argument('--no-train', action = 'store_true', help = 'do not train the network')
    parser.add_argument('--save-net', action = 'store_true', help = 'save result of traning to file')

    config = parser.parse_args()

    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)
    for handler in logger.handlers[:]:
        handler.flush()
        logger.removeHandler(handler)
    logger.addHandler(logging.StreamHandler())
    if not config.no_log and save_log:
        logger.addHandler(logging.FileHandler(config.name + '.log', mode = 'w'))

    logger.info('{}: {}'.format('name', config.name))
    logger.info('{}: {}'.format('dataset', config.dataset))
    if config.dataset != 'mnist':
        logger.info('{}: ({}, {})'.format('size', config.size[0], config.size[1]))
    logger.info('{}: {}'.format('batch_size', config.batch_size))
    logger.info('{}: {}'.format('learning rate', config.learning_rate))
    logger.info('{}: {}'.format('num epoch', config.num_epoch))

    if config.dataset == 'mnist':
        _dataset = dtstMNIST(config.size, config.batch_size)
    elif config.dataset == 'example':
        _dataset = dtstExample(config.size, config.batch_size)
    elif config.dataset == 'main':
        _dataset = dtstMain(config.size, config.batch_size)
    else:
        logger.error('unrecognized dataset')
        exit()
    
    _network = mx.sym.flatten(data=network)
    _network = mx.sym.FullyConnected(data = _network, num_hidden = _dataset['classes_num'])
    _network = mx.sym.SoftmaxOutput(data = _network, name = 'softmax')

    if not config.no_image and save_image:
        mx.viz.plot_network(_network, save_format='png',
            shape={'data':_dataset['test'].provide_data[0].shape},
            node_attrs={'shape':'rect','fixedsize':'false'}).render(config.name)

    model = mx.mod.Module(symbol = _network, context = mx.gpu())

    if not config.no_train and train:
        model.fit(
            _dataset['train'],
            eval_data = _dataset['test'],
            optimizer = 'sgd',
            optimizer_params = {'learning_rate':config.learning_rate},
            initializer = mx.initializer.Xavier(),
            eval_metric = 'acc',
            batch_end_callback = mx.callback.Speedometer(config.batch_size, 100),
            num_epoch = config.num_epoch
        )

    if not config.no_log and save_log:
        parse_log(config.name + '.log', config.name)
    
    if (config.save_net or save_network) and (not config.no_train and train):
        model.save_params(config.name + '.net')
