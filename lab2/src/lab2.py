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
}

networks = [
    {'size': (108, 72), 'network': FullyConnected([(2300,), (1150,), (575,), (288,), (144,), (72,), (36,)], default_activation = 'tanh')},
    {'size': (108, 72), 'network': FullyConnected([(2300,), (1150,), (575,), (288,), (144,), (72,), (36,)], default_activation = 'relu')},
    {'size': (108, 72), 'network': FullyConnected([(11600,), (5800,), (2900,), (1450,), (720,), (360,), (180,), (90,), (45,), (22,)], default_activation = 'relu')},
    {'size': (108, 72), 'network': FullyConnected([(2300,), (230,), (23,)], default_activation = 'relu')},
    {'size': (210, 140), 'network': FullyConnected([(8820,), (4410,), (2200,), (1100,), (550,), (270,), (135,), (68,), (34,)], default_activation = 'relu')}
]

counter = 0
for network in networks:
    fit(network, name = 'fcnn' + str(counter), **config, **network)
    counter += 1