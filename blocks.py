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