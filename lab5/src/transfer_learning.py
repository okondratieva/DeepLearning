import mxnet as mx
import os
import sys

project_path = os.path.dirname(__file__)
project_path = os.path.join(project_path, '..', '..')
sys.path.append(project_path)
from fit import save_image

def symbol():
    symbol = mx.sym.load('mysqueezenet' + '.json')
    return symbol

def load_params():
    save_dict = mx.ndarray.load('squeezenet.params')
    arg_params = {}
    aux_params = {}
    for k, value in save_dict.items():
        arg_type, name = k.split(':', 1)
        if arg_type == 'arg':
            if 'conv10' not in name:
                arg_params[name] = value
        elif arg_type == 'aux':
            if 'conv10' not in name:
                aux_params[name] = value
        else:
            raise ValueError("Invalid param file " + 'additional.params')
    return arg_params, aux_params