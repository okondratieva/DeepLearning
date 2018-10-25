import os
import shutil

current = os.path.dirname(__file__)
origin = os.path.join(current, 'dataset', 'origin')
dest = os.path.join(current, 'dataset', '432x288')

train = os.path.join(origin, 'train')
test = os.path.join(origin, 'test')
example = os.path.join(origin, 'example')

dst_train = os.path.join(dest, 'train')
dst_test = os.path.join(dest, 'test')
dst_example = os.path.join(dest, 'example')

favorites = [
    'Aquaman v7',
    'Batgirl v4',
    'Batman v2',
    'Batwing',
    'Batwoman',
    'Catwoman v4',
    'Green Arrow',
    'Green Lantern',
    'Harley Quinn',
    'Nightwing v3',
    'Wonder Woman',
    'Red Lanterns',
    'Sinestro',
    'Supergirl v6'
]

total_train = 0
total_test = 0
total_categories = 0
for category in os.listdir(train):
    if category in favorites:
        category_train = os.path.join(train, category)
        category_test = os.path.join(test, category)
        category_example = os.path.join(example, category)
        train_count = len(os.listdir(category_train))
        test_count = len(os.listdir(category_test))
        total_categories += 1
        total_train += train_count
        total_test += test_count
        print(
            train_count, " ",
            test_count, " ",
            category
        )
        shutil.copytree(category_train, os.path.join(dst_train, category))
        shutil.copytree(category_test, os.path.join(dst_test, category))
        shutil.copytree(category_example, os.path.join(dst_example, category))

print('total categories: ', total_categories)
print('total_train: ', total_train)
print('total_test: ', total_test)