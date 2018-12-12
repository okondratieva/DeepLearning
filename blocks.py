import mxnet as mx

def FullyConnected(config, input = None, default_activation = 'relu', dropout = None):
    layers = []
    for layer in config:
        if len(layer) == 1:
            layers.append((layer[0], default_activation))
        else:
            layers.append((layer[0], layer[1]))

    if input is None:
        input = mx.sym.var('data')
    output = mx.sym.flatten(data=input)

    for layer in layers:
        output = mx.sym.FullyConnected(data = output, num_hidden = layer[0])
        output = mx.sym.Activation(data = output, act_type = layer[1])
        if not (dropout is None):
            output = mx.sym.Dropout(data = output, p = dropout)

    return output

def Convolution(num, conv_size = 3, conv_stride = 1, pool_size = 2, pool_stride = 2, input = None):
    if input is None:
        input = mx.sym.var('data')
    output = mx.sym.Convolution(data = input,
        kernel = (conv_size, conv_size),
        stride = (conv_stride, conv_stride),
        num_filter = num,
        pad = (conv_size // 2, conv_size // 2))
    output = mx.sym.Activation(data = output, act_type = 'relu')
    output = mx.sym.Pooling(data = output,
        pool_type = 'max',
        kernel = (pool_size, pool_size),
        stride = (pool_stride, pool_stride))
    return output

def SmallInseption(input = None):
    if input is None:
        input = mx.sym.var('data')
    
    branches = [input for i in range(4)]

    branches[0] = mx.sym.Convolution(data = branches[0], kernel = (1, 1), stride = (1, 1), num_filter = 10)
    branches[0] = mx.sym.Activation(data = branches[0], act_type = 'relu')
    branches[0] = mx.sym.Convolution(data = branches[0], kernel = (5, 5), stride = (1, 1), num_filter = 20, pad = (2, 2))
    branches[0] = mx.sym.Activation(data = branches[0], act_type = 'relu')

    branches[1] = mx.sym.Convolution(data = branches[1], kernel = (1, 1), stride = (1, 1), num_filter = 10)
    branches[1] = mx.sym.Activation(data = branches[1], act_type = 'relu')
    branches[1] = mx.sym.Convolution(data = branches[1], kernel = (3, 3), stride = (1, 1), num_filter = 20, pad = (1, 1))
    branches[1] = mx.sym.Activation(data = branches[1], act_type = 'relu')

    branches[2] = mx.sym.Convolution(data = branches[2], kernel = (1, 1), stride = (1, 1), num_filter = 10)
    branches[2] = mx.sym.Activation(data = branches[2], act_type = 'relu')

    branches[3] = mx.sym.Pooling(data = branches[3], kernel = (3, 3), stride = (1, 1), pad = (1, 1), pool_type = 'max')
    branches[3] = mx.sym.Convolution(data = branches[3], kernel = (1, 1), stride = (1, 1), num_filter = 10)
    branches[3] = mx.sym.Activation(data = branches[3], act_type = 'relu')

    output = mx.sym.concat(*branches)

    output = mx.sym.Pooling(data = output, kernel = (2, 2), stride = (2, 2), pool_type = 'max')

    return output

def Inseption(input = None, coeff = 1, pooling = True):
    if input is None:
        input = mx.sym.var('data')
    
    branches = [input for i in range(4)]

    branches[0] = mx.sym.Convolution(data = branches[0], kernel = (1, 1), stride = (1, 1), num_filter = 10 * coeff)
    branches[0] = mx.sym.Activation(data = branches[0], act_type = 'relu')
    branches[0] = mx.sym.Convolution(data = branches[0], kernel = (5, 5), stride = (1, 1), num_filter = 20 * coeff, pad = (2, 2))
    branches[0] = mx.sym.Activation(data = branches[0], act_type = 'relu')

    branches[1] = mx.sym.Convolution(data = branches[1], kernel = (1, 1), stride = (1, 1), num_filter = 60 * coeff)
    branches[1] = mx.sym.Activation(data = branches[1], act_type = 'relu')
    branches[1] = mx.sym.Convolution(data = branches[1], kernel = (3, 3), stride = (1, 1), num_filter = 80 * coeff, pad = (1, 1))
    branches[1] = mx.sym.Activation(data = branches[1], act_type = 'relu')

    branches[2] = mx.sym.Convolution(data = branches[2], kernel = (1, 1), stride = (1, 1), num_filter = 40 * coeff)
    branches[2] = mx.sym.Activation(data = branches[2], act_type = 'relu')

    branches[3] = mx.sym.Pooling(data = branches[3], kernel = (3, 3), stride = (1, 1), pad = (1, 1), pool_type = 'max')
    branches[3] = mx.sym.Convolution(data = branches[3], kernel = (1, 1), stride = (1, 1), num_filter = 20 * coeff)
    branches[3] = mx.sym.Activation(data = branches[3], act_type = 'relu')

    output = mx.sym.concat(*branches)

    if pooling:
        output = mx.sym.Pooling(data = output, kernel = (2, 2), stride = (2, 2), pool_type = 'max')

    return output