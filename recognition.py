import os
import pickle
import numpy as np
import cv2.cv2 as cv2
import tensorflowjs as tfjs
from tensorflow.python.keras import activations
from tensorflow.python.keras import Sequential, layers
from tensorflow.python.keras.models import load_model
from keras.preprocessing.text import Tokenizer
from keras.utils.np_utils import to_categorical
from utils import sort_contours, sigmoid
import pytesseract

pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

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
        self._model = Sequential()
        self.tokenizer = Tokenizer(char_level=True,lower=False)

    def _load_tokenizer(self, path):
        with open(os.path.join(path, self.model_name, 'tokenizer.pickle'), 'rb') as file:
            self.tokenizer = pickle.load(file)

    @classmethod
    def load(cls, model_name: str, path: str = '.', summery=False):
        """Load model"""
        model = cls(model_name=model_name)

        model.model = load_model(os.path.join(path, model_name))
        model._model = load_model(os.path.join(path, model_name))
        model._model.layers[-1].activation = activations.get('tanh')
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
    def _get_layers(input_shape, output_shape, output_activation='softmax'):
        return [
            layers.Conv2D(32, (3, 3), padding='same', activation='relu', input_shape=input_shape),
            layers.MaxPooling2D((2, 2), strides=2),
            layers.Conv2D(64, (3, 3), padding='same', activation='relu'),
            layers.MaxPooling2D((2, 2), strides=2),
            layers.Flatten(),
            layers.Dense(128, activation='relu'),
            layers.Dropout(0.3),
            layers.Dense(64, activation='relu'),
            layers.Dropout(0.2),
            layers.Dense(output_shape, activation=output_activation, name='Output')
        ]

    def train(self, x_train: np.array, y_train: np.array, batch_size: int = 32, epochs: int = 5,
              validation_split: float = 0.2, save: bool = True, save_path: str = '.', save_js=False, summery=False):
        """Train model and save"""
        self.tokenizer.fit_on_texts(''.join(str(y) for y in set(y_train)))
        y_train = self._letter_list_to_categorical(y_train)

        input_shape = x_train[0].shape
        output_shape = len(y_train[0])
        self.model = Sequential(self._get_layers(input_shape, output_shape, 'softmax'))

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

    def predict(self, x: np.array) -> tuple[str, float]:
        """Predict and return letter"""
        x = np.expand_dims(x, axis=0)
        y = self._model.predict(x)[0]
        index = np.argmax(y, 0)
        letter = self.tokenizer.index_word[index] if not index == 0 else ' '
        return letter, y[index]

    def predicts(self, x_list: np.array) -> list[str]:
        """Predict and return list of letter"""
        y = self._model.predict(x_list)
        indexs = np.argmax(y, 1)
        return np.array([self.tokenizer.index_word[i] if not i == 0 else ' ' for i in indexs])


def image_to_string(image, model):
    text = ''
    eps = []
    h = image.shape[0]
    w = image.shape[1]
    image = cv2.resize(image, (w * 2, h * 2))
    _, image = cv2.threshold(image, 200, 255, cv2.THRESH_BINARY)
    contours, tree = cv2.findContours(image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    for contour in sort_contours(contours, 1):
        x, y, w, h = cv2.boundingRect(contour)
        if w > 15 and h > 15:
            img = image[y: y + h, x: x + w]
            if w >= 2 * h:
                k = w // (h * 0.8)
                while True:
                    try:
                        letter_images = np.hsplit(img, k)
                        for letter_image in letter_images:
                            t, e = image_to_letter(letter_image, model)
                            text += t
                            eps.append(e)
                        break
                    except ValueError:
                        img = img[0: h, 0: -1]
                        continue
            else:
                t, e = image_to_letter(img, model)
                text += t
                eps.append(e)
        else:
            text += ''
    if not eps:
        eps = [0]
    return text, sum(eps) / len(eps)


# def image_to_string(image, model, _eps=0.7):
#     _, image = cv2.threshold(image, 200, 255, cv2.THRESH_BINARY)
#     image = cv2.copyMakeBorder(image, 10, 10, 10, 10, cv2.BORDER_CONSTANT)
#     data = pytesseract.image_to_data(image, lang='rus+eng', config=r'--oem 3 --psm 6',
#                                      output_type=pytesseract.Output.DICT)
#     eps_n = 0
#     eps = []
#     for c in data['conf']:
#         c = float(c)
#         if c > 0:
#             eps.append(c)
#     try:
#         eps = sum(eps) / len(eps)
#     except ZeroDivisionError:
#         eps = 0
#     if eps < _eps * 100:
#         text, eps_n = _image_to_string(image, model)
#     else:
#         text = ''.join(data['text'])
#     return {'text': text, 'eps': eps, 'eps_n': eps_n}


def image_to_letter(image, model: LetterRecognition):
    image = cv2.resize(image, (100, 100))
    # image = cv2.morphologyEx(image, cv2.MORPH_DILATE, np.ones((3, 3), np.uint8))
    image = cv2.copyMakeBorder(image, 30, 30, 15, 15, cv2.BORDER_CONSTANT)
    image = cv2.resize(image, (28, 28))
    image = np.expand_dims(image, axis=2)
    text = model.predict(image)
    return text
