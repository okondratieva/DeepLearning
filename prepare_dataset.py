import mxnet as mx
import subprocess
import os

def dataset(name, path):
    return {'name': name, 'path': path}

file_path = os.path.dirname(__file__)
dataset_root = os.path.join(file_path, 'dataset')
output_path = os.path.join(dataset_root, 'pack')
im2rec_path = mx.test_utils.get_im2rec_path()

dataset_types = [
    'example',
    'train',
    'test'
]

for size in os.listdir(dataset_root):
    if str(size) != 'pack' and str(size) != 'origin':
        name = dataset_types[0] + '_' + size

        #example dataset preparation
            #generate .lst file
        subprocess.check_call(
            [
                'python', im2rec_path,
                '--list',
                '--recursive',
                '--num-thread', '4',
                '--train-ratio', '0.8',
                os.path.join(output_path, name),
                os.path.join(dataset_root, size, dataset_types[0])
            ],
            stderr=subprocess.STDOUT
        )
            #generate binary and .idx files
        subprocess.check_call(
            [
                'python', im2rec_path,
                '--num-thread', '4',
                '--pass-through',
                os.path.join(output_path, name),
                os.path.join(dataset_root, size, dataset_types[0])
            ],
            stderr=subprocess.STDOUT
        )

        #main dataset preparation
        for dataset_type in dataset_types[1:]:
            name = dataset_type + '_' + size
             #generate .lst file
            subprocess.check_call(
                [
                    'python', im2rec_path,
                    '--list',
                    '--recursive',
                    '--num-thread', '4',
                    os.path.join(output_path, name),
                os.path.join(dataset_root, size, dataset_type)
                ],
                stderr=subprocess.STDOUT
            )
            #generate binary and .idx files
            subprocess.check_call(
                [
                    'python', im2rec_path,
                    '--num-thread', '4',
                    '--pass-through',
                    os.path.join(output_path, name),
                    os.path.join(dataset_root, size, dataset_type)
                ],
                stderr=subprocess.STDOUT
            )