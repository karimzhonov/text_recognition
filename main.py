import typer
import imutils
import cv2.cv2 as cv2
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from dataset import load_dataset
from utils import pdf2numpy, recurent_findContours, sort_contours
from recognition import LetterRecognition, image_to_string

app = typer.Typer()


@app.command()
def main(mode: str = '2'):
    if mode == '1':
        images = pdf2numpy('src/doc.pdf')
        img = images[0]
        height, weight, _ = img.shape
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        _, img_for_findContours = cv2.threshold(img, 160, 255, cv2.THRESH_BINARY_INV)
        img_for_findContours = imutils.rotate(img_for_findContours, 3)
        img_for_findContours = cv2.bilateralFilter(img_for_findContours, 3, 0, 0)
        _, img_for_findContours = cv2.threshold(img_for_findContours, 30, 255, cv2.THRESH_BINARY_INV)
        cv2.imshow('Image',
                   cv2.rotate(cv2.resize(img_for_findContours, (weight // 2, height // 2)), cv2.ROTATE_90_CLOCKWISE))
        contours = recurent_findContours(img_for_findContours, weight // 2, 30, 30)
        # Tensorflow model
        recognition = LetterRecognition.load('model_cyrillic_numbers_100epochs')

        for word in contours:
            word_image = word['image']
            x0, y0 = word['x'], word['y']
            for letter in recurent_findContours(word_image, 100, 10, 10, x0, y0, output_shape=(28, 28)):
                x, y = letter['x'], letter['y']
                letter_image = letter['image']
                letter_image = cv2.copyMakeBorder(letter_image, 8, 8, 8, 8, cv2.BORDER_CONSTANT)
                letter_image = cv2.resize(letter_image, (28, 28))
                letter_image = np.expand_dims(letter_image, axis=2)
                text = recognition.predict(letter_image)
                print(f'{text}', end='')
                cv2.imshow('Letter', letter_image)
                cv2.waitKey(0)
            print()
        while not cv2.waitKey(0) == ord('q'): break
    elif mode == '2':
        image_list = []
        images = pdf2numpy('src/doc.pdf')
        model = LetterRecognition.load('model_cyrillic_numbers_100epochs')
        for img in images[:3]:
            height, weight, _ = img.shape
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            _, img_for_findContours = cv2.threshold(img, 180, 255, cv2.THRESH_BINARY_INV)
            # Rotate angle
            contours, _ = cv2.findContours(img_for_findContours, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
            max_contour = contours[0]
            max_contour_area = cv2.contourArea(max_contour)
            for contour in contours:
                contour_area = cv2.contourArea(contour)
                if contour_area > max_contour_area:
                    max_contour_area = contour_area
                    max_contour = contour
            _, _, angle = cv2.minAreaRect(max_contour)
            # Delete lines
            img_for_HoughLinesP = cv2.GaussianBlur(img, (3, 3), 10)
            _, img_for_HoughLinesP = cv2.threshold(img_for_HoughLinesP, 240, 255, cv2.THRESH_BINARY_INV)
            ver_lines = cv2.HoughLinesP(img_for_HoughLinesP, 1, np.pi, 100, minLineLength=50)
            hor_lines = cv2.HoughLinesP(img_for_HoughLinesP, 1, np.pi / 180, 200, minLineLength=100)
            for [[x0, y0, x1, y1]] in [*ver_lines, *hor_lines]:
                img_for_findContours = cv2.line(img_for_findContours, (x0, y0), (x1, y1), (0, 0, 0), 3)

            img_for_findContours = imutils.rotate(img_for_findContours, angle)

            img_for_findContours = cv2.bilateralFilter(img_for_findContours, 11, 100, 250)
            _, img_for_findContours = cv2.threshold(img_for_findContours, 100, 255, cv2.THRESH_BINARY)
            img_for_findContours = cv2.bilateralFilter(img_for_findContours, 9, 100, 100)
            contours, tree = cv2.findContours(img_for_findContours, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
            cv2.imshow('res', cv2.resize(img_for_findContours, (weight // 3, height // 3)))

            img_for_findContours_show = cv2.cvtColor(img_for_findContours, cv2.COLOR_GRAY2BGR)
            for c in sort_contours(contours, 0):
                x, y, w, h = cv2.boundingRect(c)
                if w > 15 and h > 15:
                    image = img_for_findContours[y: y + h, x: x + w]
                    text = image_to_string(image, model)
                    image_list.append({
                        'x': x, 'y': y,
                        'image': image,
                        'text': text
                    })
                    cv2.imshow('text', image)
                    cv2.waitKey(0)
                    cv2.destroyAllWindows()
                    cv2.rectangle(img_for_findContours_show, (x, y), (x + w, y + h), (0, 0, 255), 2)

            cv2.waitKey(0)


@app.command()
def train():
    recognition = LetterRecognition(model_name='model_cyrillic_numbers_50epochs')
    x, y = load_dataset('numbers', 'cyrillic')
    x_train, x_test, y_train, y_test = train_test_split(x, y)
    history = recognition.train(x_train, y_train, epochs=100, summery=True)
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('Train')
    plt.show()


@app.command()
def test():
    recognition = LetterRecognition.load('model_cyrillic_numbers_100epochs')
    x, y = load_dataset('numbers', 'cyrillic')
    x_train, x_test, y_train, y_test = train_test_split(x, y)
    predicts = recognition.predicts(x_test)
    mask = predicts == y_test
    x_false = x_test[~mask]
    print(f'{x_false.shape[0]}/{x_test.shape[0]}')
    # Evalute
    recognition.evaluate(x_test, y_test, 32)


if __name__ == '__main__':
    app()
