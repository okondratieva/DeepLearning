import mxnet as mx
import logging
import os

file_path = os.path.dirname(__file__)
pack_path = os.path.join(file_path, 'dataset', 'pack')

def dataset(train, test, classes_num):
    return {'train':train, 'test':test, 'classes_num':classes_num}

def dtstMNIST(shape, batch_size):
    mnist = mx.test_utils.get_mnist()
    return dataset(
        mx.io.NDArrayIter(mnist['train_data'], mnist['train_label'], batch_size, shuffle=True),
        mx.io.NDArrayIter(mnist['test_data'], mnist['test_label'], batch_size),
        10
    )

def dtstExample(shape, batch_size):
    size = str(shape[0]) + 'x' + str(shape[1])
    train = mx.io.ImageRecordIter(
        path_imgrec = os.path.join(pack_path, 'example_{}_train.rec'.format(size)),
        path_imgidx = os.path.join(pack_path, 'example_{}_train.idx'.format(size)),
        data_shape = (3, shape[0], shape[1]),
        batch_size = batch_size,
        shuffle = True
    )
    test = mx.io.ImageRecordIter(
        path_imgrec = os.path.join(pack_path, 'example_{}_val.rec'.format(size)),
        path_imgidx = os.path.join(pack_path, 'example_{}_val.idx'.format(size)),
        data_shape = (3, shape[0], shape[1]),
        batch_size = batch_size
    )
    return dataset(train, test, 14)

def dtstMain(shape, batch_size):
    size = str(shape[0]) + 'x' + str(shape[1])
    train = mx.io.ImageRecordIter(
        path_imgrec = os.path.join(pack_path, 'train_{}.rec'.format(size)),
        path_imgidx = os.path.join(pack_path, 'train_{}.idx'.format(size)),
        data_shape = (3, shape[0], shape[1]),
        batch_size = batch_size,
        shuffle = True
    )
    test = mx.io.ImageRecordIter(
        path_imgrec = os.path.join(pack_path, 'test_{}.rec'.format(size)),
        path_imgidx = os.path.join(pack_path, 'test_{}.idx'.format(size)),
        data_shape = (3, shape[0], shape[1]),
        batch_size = batch_size
    )
    return dataset(train, test, 14)