import cv2
import os

file_path = os.path.dirname(__file__)
input_path = os.path.join(file_path, 'dataset', '432x288')

sizes = [
    (174, 174),
    (206, 206),
    (256, 256),
    (128, 128),
    (100, 100),
    (210, 140),
    (108, 72)
]

for size in sizes:
    output_path = os.path.join(file_path, 'dataset', str(size[0]) + 'x' + str(size[1]))

    if not os.path.exists(output_path):
        for data_type in os.listdir(input_path):
            type_path = os.path.join(input_path, data_type)
            for category in os.listdir(type_path):
                category_path = os.path.join(type_path, category)
                category_output_path = os.path.join(output_path, data_type, category)
                os.makedirs(category_output_path)
                for img in os.listdir(category_path):
                    img_path = os.path.join(category_path, img)
                    img_output_path = os.path.join(category_output_path, img)
                    image = cv2.imread(img_path)
                    image = cv2.resize(image, (size[1], size[0]), interpolation = cv2.INTER_CUBIC)
                    print(cv2.imwrite(img_output_path, image), ' ', size, ' ', img_output_path)