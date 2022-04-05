import os
import pickle
import numpy as np
import cv2.cv2 as cv2
import tensorflowjs as tfjs
from tensorflow.python.keras import Sequential, layers
from tensorflow.python.keras.models import load_model
from keras.preprocessing.text import Tokenizer
from keras.utils.np_utils import to_categorical
from utils import sort_contours


class LetterRecognition:
    def __init__(self, model_name: str = 'model_1',
                 optimizer: str = 'adam', loss: str = 'categorical_crossentropy', metric: str = 'accuracy'):
        """
        Letter Recognition
        :param model_name: Model name to save
        :param optimizer: Sequential optimizer
        :param loss: Sequential loss
        :param metric: Sequential metric
        """
        self.optimizer = optimizer
        self.loss = loss
        self.metric = metric
        self.model_name = model_name
        self.model = Sequential()
        self.tokenizer = Tokenizer(char_level=True,lower=False)

    def _load_tokenizer(self, path):
        with open(os.path.join(path, self.model_name, 'tokenizer.pickle'), 'rb') as file:
            self.tokenizer = pickle.load(file)

    @classmethod
    def load(cls, model_name: str, path: str = '.', summery=False):
        """Load model"""
        model = cls(model_name=model_name)

        model.model = load_model(os.path.join(path, model_name))
        model._load_tokenizer(path)
        if summery:
            model.model.summary()
        return model

    def _save_keras_model_as_json(self, path):
        tfjs.converters.save_keras_model(self.model, os.path.join(path, f'{self.model_name}/js/'))

    def _save_tokenizer(self, path):
        with open(os.path.join(path, self.model_name, 'tokenizer.pickle'), 'wb') as file:
            pickle.dump(self.tokenizer, file)

    def save(self, path: str = '.', save_js: bool = False):
        """Save model"""
        self.model.save(os.path.join(path, self.model_name))
        self._save_tokenizer(path)
        if save_js:
            self._save_keras_model_as_json(path)

    def _letter_list_to_categorical(self, y_train):
        y_train = np.array([self.tokenizer.texts_to_sequences(str(letter))[0] for letter in y_train])
        return to_categorical(y_train, len(self.tokenizer.word_index) + 1)

    @staticmethod
    def _get_layers(input_shape, output_shape):
        return [
            layers.Conv2D(32, (3, 3), padding='same', activation='relu', input_shape=input_shape),
            layers.MaxPooling2D((2, 2), strides=2),
            layers.Conv2D(64, (3, 3), padding='same', activation='relu'),
            layers.MaxPooling2D((2, 2), strides=2),
            layers.Flatten(),
            layers.Dense(512, activation='relu'),
            layers.Dropout(0.3),
            # layers.Dense(256, activation='relu'),
            # layers.Dropout(0.4),
            # layers.Dense(128, activation='relu'),
            # layers.Dropout(0.3),
            layers.Dense(output_shape, activation='softmax'),
        ]

    def train(self, x_train: np.array, y_train: np.array, batch_size: int = 32, epochs: int = 5,
              validation_split: float = 0.2, save: bool = True, save_path: str = '.', save_js=False, summery=False):
        """Train model and save"""
        self.tokenizer.fit_on_texts(''.join(y_train))
        y_train = self._letter_list_to_categorical(y_train)

        input_shape = x_train[0].shape
        output_shape = len(y_train[0])
        self.model = Sequential(self._get_layers(input_shape, output_shape))

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


def image_to_string(image, model):
    _, image = cv2.threshold(image, 230, 255, cv2.THRESH_BINARY)
    text = ''
    contours, tree = cv2.findContours(image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    for contour in sort_contours(contours, 1):
        x, y, w, h = cv2.boundingRect(contour)
        if w > 10 and h > 10:
            img = image[y: y + h, x: x + w]
            if w >= 2*h:
                k = w // (h * 0.8)
                while True:
                    try:
                        letter_images = np.hsplit(image, k)
                        text += ''.join([image_to_letter(letter_image, model) for letter_image in letter_images])
                        break
                    except ValueError:
                        image = image[0: h, 0: -1]
                        continue
            else:
                text += image_to_letter(img, model)
    return text


def image_to_letter(image, model: LetterRecognition):
    image = cv2.resize(image, (28, 28))
    image = cv2.copyMakeBorder(image, 8, 8, 8, 8, cv2.BORDER_CONSTANT)
    image = cv2.resize(image, (28, 28))
    image = np.expand_dims(image, axis=2)
    return model.predict(image)
