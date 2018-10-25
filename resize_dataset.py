import cv2
import os

file_path = os.path.dirname(__file__)
dataset_size = 'reduce' #'origin'
input_path = os.path.join(file_path, 'dataset', dataset_size)

img_width = 256

output_path = os.path.join(file_path, 'dataset', 'resize_' + str(img_width))

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
            image = cv2.resize(image, (img_width, img_width), interpolation = cv2.INTER_CUBIC)
            print(img_output_path, " ", cv2.imwrite(img_output_path, image))