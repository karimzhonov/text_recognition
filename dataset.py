import os
import io
import cv2.cv2 as cv2
import numpy as np
from tqdm import tqdm
from PIL import Image
from keras.datasets import mnist


def load_cyrillic(path = 'src/dataset/Cyrillic'):
    """Load Cirillic dataset"""
    print(f'[+] Init dataset: {path}')
    x_train, y_train = [], []
    for i in tqdm(os.listdir(path)):
        for j in os.listdir(os.path.join(path, i)):
            image_path = os.path.join(path, i, j)
            with open(image_path, 'rb') as file:
                byte = io.BytesIO(file.read())
                image = Image.open(byte)
                image = np.array(image, dtype=np.uint8)
                image = cv2.cvtColor(image, cv2.COLOR_BGRA2GRAY)
                image = cv2.resize(image, (28, 28))
                image = np.expand_dims(image / 255, 2)
                x_train.append(image)
                y_train.append(i.lower())
    x_train = np.array(x_train)
    return x_train, y_train


def load_numbers():
    """Load numbers dataset"""
    print('[+] Init dataset: mnist')
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    x_train = np.concatenate([x_train, x_test])
    y_train = np.concatenate([y_train, y_test])
    x_train = np.expand_dims(x_train / 255, axis=3)
    return x_train, y_train


def load_dataset(*dataset_list):
    """
    Load daatset
    :param dataset_list: 'cyrillic', 'numbers'
    :return: list of image and list of letter or numbers
    """
    data_list = ['cyrillic', 'numbers']
    x_train, y_train = [], []
    for dataset_name in dataset_list:
        if not dataset_name in data_list:
            raise ValueError(f'Dataset name must be one of the ({data_list}), not {dataset_name}')
        lx_train, ly_train = eval(f'load_{dataset_name}')()
        x_train.append(lx_train)
        y_train.append(ly_train)
    x_train = np.concatenate(x_train)
    y_train = np.concatenate(y_train)
    return x_train, y_train
