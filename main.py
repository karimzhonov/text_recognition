import os
import typer
import imutils
import cv2.cv2 as cv2
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from dataset import load_dataset
from utils import pdf2numpy, recurent_findContours
from recognition import LetterRecognition

app = typer.Typer()
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'


@app.command()
def main():
    images = pdf2numpy('src/doc.pdf')
    img = images[1]
    height, weight, _ = img.shape
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, img_for_findConoturs = cv2.threshold(img, 160, 255, cv2.THRESH_BINARY_INV)
    img_for_findConoturs = imutils.rotate(img_for_findConoturs, 3)
    img_for_findConoturs = cv2.bilateralFilter(img_for_findConoturs, 5, 0, 0)
    _, img_for_findContours = cv2.threshold(img_for_findConoturs, 30, 255, cv2.THRESH_BINARY_INV)
    cv2.imshow('Image',
               cv2.rotate(cv2.resize(img_for_findConoturs, (weight // 2, height // 2)), cv2.ROTATE_90_CLOCKWISE))
    contours = recurent_findContours(img_for_findContours, weight // 2, 30, 30)
    # Tensorflow model
    recognition = LetterRecognition.load('model_1')

    # word = contours[4]
    for word in contours:
        word_image = word['image']
        x0, y0 = word['x'], word['y']
        for letter in recurent_findContours(word_image, 100, 10, 10, x0, y0, output_shape=(28, 28)):
            x, y = letter['x'], letter['y']
            letter_image = letter['image']
            letter_image = np.expand_dims(letter_image, axis=2)
            text = recognition.predict(letter_image)
            print(f'{text} --- {x, y}')
            cv2.imshow('Letter', letter_image)
            cv2.waitKey(0)
    while not cv2.waitKey(0) == ord('q'): break


@app.command()
def train():
    recognition = LetterRecognition(model_name='model_1', output_shape=11)
    x, y = load_dataset('numbers')
    x_train, x_test, y_train, y_test = train_test_split(x, y)
    history = recognition.train(x_train, y_train, epochs=5)
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('Train')
    plt.show()


@app.command()
def test():
    recognition = LetterRecognition.load('model_1')
    x, y = load_dataset('numbers')
    x_train, x_test, y_train, y_test = train_test_split(x, y)
    predicts = recognition.predicts(x_test)
    mask = predicts == y_test
    x_false = x_test[~mask]
    print(f'{x_false.shape[0]}/{x_test.shape[0]}')
    # Evalute
    recognition.evaluate(x_test, y_test, 32)


if __name__ == '__main__':
    app()
