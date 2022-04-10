import numpy as np
import pandas as pd
from tqdm import tqdm
from keras.datasets import mnist


def load_numbers(numbers_count=None, **kwargs):
    """Load numbers dataset"""
    print('[+] Init dataset: mnist')
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    x_train = np.concatenate([x_train, x_test])
    y_train = np.concatenate([y_train, y_test])
    if numbers_count:
        x_train = x_train[:numbers_count]
        y_train = y_train[:numbers_count]
    return x_train, y_train


def load_latin(latin_count = None, **kwargs):
    """Load latin dataset"""
    print('[+] Init dataset: ')


def load_cyrillic(cyrillic_count=None, **kwargs):
    """Load cyrillic dataset, count - 345500"""
    print(f'[+] Init dataset: HMCC balanced')
    symbols = 'АаБбВвГгДдЕеЁёЖжЗзИийКкЛлМмНнОоОоПпРрСсТтУуYyФфХхЦцЧчШшЩщъыьЭэЮюЯя'
    images = pd.read_csv('src/dataset/HMCC balanced.csv', delimiter=',', nrows=cyrillic_count)
    y_train, x_train = [], []
    for i in tqdm(range(images.shape[0])):
        image = images.iloc[i]
        image = np.array(image, dtype=np.uint8)
        y = symbols[int(image[0])]
        x = image[1:].reshape(28, 28)
        y_train.append(y)
        x_train.append(x)
    return np.array(x_train), np.array(y_train)


def load_dataset(*datasets, **params):
    """
    Load daatset
    :param datasets: 'cyrillic', 'numbers'
    :return: list of image and list of letter or numbers
    """
    data_list = ['numbers', 'cyrillic']
    x_train, y_train = [], []
    for dataset_name in datasets:
        if not dataset_name in data_list:
            raise ValueError(f'Dataset name must be one of the {data_list}, not {dataset_name}')
        lx_train, ly_train = eval(f'load_{dataset_name}')(**params)
        x_train.append(lx_train)
        y_train.append(ly_train)
    x_train = np.concatenate(x_train)
    y_train = np.concatenate(y_train)
    print(f'[+] Dataset ready: x - {x_train.shape}, y - {y_train.shape}')
    return x_train, y_train
