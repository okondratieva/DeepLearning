import mxnet as mx

def FullyConnected(config, input = None, default_activation = 'relu'):
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