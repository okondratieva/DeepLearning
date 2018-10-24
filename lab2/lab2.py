import mxnet as mx
import os

file_path = os.path.dirname(__file__)
dataset_size = 'reduce' #'origin'
dataset_calsses_num = 14 #86
pack_root = os.path.join(file_path, '..', 'dataset', 'pack', dataset_size)
shape = (3, 432, 288)

batch_size = 1

def dataset(train, test, classes_num):
    return {'train':train, 'test':test, 'classes_num':classes_num}

def dtstMNIST():
    mnist = mx.test_utils.get_mnist()
    return dataset(
        mx.io.NDArrayIter(mnist['train_data'], mnist['train_label'], batch_size, shuffle=True),
        mx.io.NDArrayIter(mnist['test_data'], mnist['test_label'], batch_size),
        10
    )

def dtstExample():
    train = mx.io.ImageRecordIter(
        path_imgrec = os.path.join(pack_root, 'example_train.rec'),
        path_imgidx = os.path.join(pack_root, 'example_train.idx'),
        data_shape = shape,
        batch_size = batch_size
    )
    test = mx.io.ImageRecordIter(
        path_imgrec = os.path.join(pack_root, 'example_val.rec'),
        path_imgidx = os.path.join(pack_root, 'example_val.idx'),
        data_shape = shape,
        batch_size = batch_size
    )
    return dataset(train, test, dataset_calsses_num)

def dtstMain():
    train = mx.io.ImageRecordIter(
        path_imgrec = os.path.join(pack_root, 'train.rec'),
        path_imgidx = os.path.join(pack_root, 'train.idx'),
        data_shape = shape,
        batch_size = batch_size
    )
    test = mx.io.ImageRecordIter(
        path_imgrec = os.path.join(pack_root, 'test.rec'),
        path_imgidx = os.path.join(pack_root, 'test.idx'),
        data_shape = shape,
        batch_size = batch_size
    )
    return dataset(train, test, dataset_calsses_num)