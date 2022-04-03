import os
import sys
import pickle
import numpy as np
import tensorflow as tf
import tensorflowjs as tfjs
from tensorflow.python import keras
from tensorflow.python.keras import Sequential, layers
from tensorflow.python.keras.models import load_model
from keras.preprocessing.text import Tokenizer
from keras.utils.np_utils import to_categorical


class LetterRecognition:
    def __init__(self, input_shape: tuple[int] = (28, 28, 1), output_shape: int = 35 + 10, model_name: str = 'model_1',
                 optimizer: str = 'adam', loss: str = 'categorical_crossentropy', metric: str = 'accuracy',
                 conv2d_activation: str = 'relu', conv2d_padding: str = 'same', dense_activation: str = 'relu',
                 out_activation: str = 'softmax'):
        """
        Letter Recognition
        :param input_shape: tuple[h, w, c]
        :param output_shape: count out variants
        :param model_name: Model name to save
        :param optimizer: Sequential optimizer
        :param loss: Sequential loss
        :param metric: Sequential metric
        :param conv2d_activation: Conv2d layer activation
        :param conv2d_padding:  Conv2d layer padding
        :param dense_activation: Dense layer activation
        :param out_activation: Out layer activation
        """
        self.input_shape = input_shape
        self.output_shape = output_shape
        self.optimizer = optimizer
        self.loss = loss
        self.metric = metric
        self.model_name = model_name
        self.layers = [
            layers.Conv2D(32, (3, 3), padding=conv2d_padding, activation=conv2d_activation, input_shape=input_shape),
            layers.MaxPooling2D((2, 2), strides=2),
            layers.Conv2D(64, (3, 3), padding=conv2d_padding, activation=conv2d_activation),
            layers.MaxPooling2D((2, 2), strides=2),
            layers.Flatten(),
            layers.Dense(256, activation=dense_activation),
            layers.Dropout(0.4),
            layers.Dense(128, activation=dense_activation),
            layers.Dropout(0.4),
            layers.Dense(output_shape, activation=out_activation),
        ]
        self.model = Sequential(self.layers)
        self.tokenizer = Tokenizer(char_level=True)
        self._version_as_console()

    @staticmethod
    def _version_as_console():
        print(f'[+] Python - {sys.version}')
        print(f'[+] Tenseflow - {tf.__version__}')
        print(f'[+] Keras - {keras.__version__}')

    @classmethod
    def load(cls, model_name: str, path: str = '.', summery=False):
        """Load model"""
        model = cls(model_name=model_name)

        model.model = load_model(os.path.join(path, model_name))
        model._load_tokenizer(path)
        if summery:
            model.model.summary()
        return model

    def save(self, path: str = '.', save_js: bool = False):
        """Save model"""
        self.model.save(os.path.join(path, self.model_name))
        self._save_tokenizer(path)
        if save_js:
            self._save_keras_model_as_json(path)

    def _save_keras_model_as_json(self, path):
        tfjs.converters.save_keras_model(self.model, os.path.join(path, f'{self.model_name}/js/'))

    def _save_tokenizer(self, path):
        with open(os.path.join(path, self.model_name, 'tokenizer.pickle'), 'wb') as file:
            pickle.dump(self.tokenizer, file)

    def _load_tokenizer(self, path):
        with open(os.path.join(path, self.model_name, 'tokenizer.pickle'), 'rb') as file:
            self.tokenizer = pickle.load(file)

    def _letter_list_to_categorical(self, y_train):
        symbols = ''.join([str(l) for l in set(y_train)])
        self.tokenizer.fit_on_texts(symbols)
        y_train = np.array([self.tokenizer.texts_to_sequences(str(letter).lower())[0] for letter in y_train])
        return to_categorical(y_train, len(symbols) + 1)

    def train(self, x_train: np.array, y_train: np.array, batch_size: int = 32, epochs: int = 5,
              validation_split: float = 0.2, save: bool = True, save_path: str = '.', save_js=False, summery=False):
        """Train model and save"""
        y_train = self._letter_list_to_categorical(y_train)

        if summery:
            self.model.summary()
        self.model.compile(
            optimizer=self.optimizer,
            loss=self.loss,
            metrics=[self.metric],
        )
        history = self.model.fit(x_train, y_train, batch_size=batch_size, epochs=epochs,
                                 validation_split=validation_split)
        if save:
            self.save(save_path, save_js)
        return history

    def evaluate(self, x: np.array, y: np.array, batch_size: int = 1,
                 save: bool = True, save_path: str = '.', save_js: bool = False):
        """Evaluate model and save"""
        y = self._letter_list_to_categorical(y)
        history = self.model.evaluate(x, y, batch_size)
        if save:
            self.save(save_path, save_js)
        return history

    def predict(self, x: np.array) -> str:
        """Predict and return letter"""
        x = np.expand_dims(x, axis=0)
        return self.predicts(x)[0]

    def predicts(self, x_list: np.array) -> list[str]:
        """Predict and return list of letter"""
        y = self.model.predict(x_list)
        indexs = np.argmax(y, 1)
        return np.array([self.tokenizer.index_word[i] if not i == 0 else ' ' for i in indexs])
