import os
import numpy as np
import cv2

from .hyperparameters import dataset_path

def representative_data_gen():
    for file in os.listdir(dataset_path):
        image_path = dataset_path + file
        input_data = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        input_data = cv2.resize(input_data, (200, 31))
        input_data = input_data[np.newaxis]
        input_data = np.expand_dims(input_data, 3)
        input_data = input_data.astype('float32')/255
        yield [input_data]