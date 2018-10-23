import mxnet as mx
import subprocess
import os

def dataset(name, path):
    return {'name': name, 'path': path}

file_path = os.path.dirname(__file__)
dataset_root = os.path.join(file_path, 'dataset')
output_path = os.path.join(dataset_root, 'pack')

datasets = [
    dataset('example', os.path.join(dataset_root, 'example_dataset')),
    dataset('train', os.path.join(dataset_root, 'train')),
    dataset('test', os.path.join(dataset_root, 'test'))
]

im2rec_path = mx.test_utils.get_im2rec_path()

#example dataset preparation
    #generate .lst file
subprocess.check_call(
    [
        'python', im2rec_path,
        '--list',
        '--recursive',
        '--num-thread', '4',
        '--train-ratio', '0.8',
        os.path.join(output_path, datasets[0]['name']),
        datasets[0]['path']
    ],
    stderr=subprocess.STDOUT
)
    #generate binary and .idx files
subprocess.check_call(
    [
        'python', im2rec_path,
        '--num-thread', '4',
        '--pass-through',
        os.path.join(output_path, datasets[0]['name']),
        datasets[0]['path']
    ],
    stderr=subprocess.STDOUT
)

#main dataset preparation
for dataset in datasets[1:]:
    subprocess.check_call(
    [
        'python', im2rec_path,
        '--list',
        '--recursive',
        '--num-thread', '4',
        os.path.join(output_path, dataset['name']),
        dataset['path']
    ],
    stderr=subprocess.STDOUT
)
    #generate binary and .idx files
subprocess.check_call(
    [
        'python', im2rec_path,
        '--num-thread', '4',
        '--pass-through',
        os.path.join(output_path, dataset['name']),
        dataset['path']
    ],
    stderr=subprocess.STDOUT
) 