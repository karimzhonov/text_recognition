import typer
import imutils
import cv2.cv2 as cv2
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from dataset import load_dataset
from utils import pdf2numpy, recurent_findContours, sort_contours, find_rotate_angle
from recognition import LetterRecognition, image_to_string
import pytesseract

pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'
app = typer.Typer()


@app.command()
def main(mode: str = '2'):
    images = pdf2numpy('src/doc.pdf')
    # Tensorflow model
    recognition = LetterRecognition.load('model_numbers_2dense_128_64')
    if mode == '1':
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

        for word in contours:
            word_image = word['image']
            x0, y0 = word['x'], word['y']
            for letter in recurent_findContours(word_image, 100, 10, 10, x0, y0, output_shape=(28, 28)):
                x, y = letter['x'], letter['y']
                letter_image = letter['image']
                letter_image = cv2.copyMakeBorder(letter_image, 8, 8, 8, 8, cv2.BORDER_CONSTANT)
                letter_image = cv2.resize(letter_image, (28, 28))
                letter_image = np.expand_dims(letter_image, axis=2)
                cv2.imshow('letter', letter_image)
                cv2.waitKey(0)
                text = recognition.predict(letter_image)
                print(f'{text}', end='')
            print()
            cv2.imshow('word', 255 - word_image)
            cv2.waitKey(0)
    elif mode == '2':
        for img in images:
            image_list = []
            height, weight, _ = img.shape
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            _, img_for_findContours = cv2.threshold(img, 180, 255, cv2.THRESH_BINARY_INV)
            # Rotate angle
            angle = find_rotate_angle(img_for_findContours)
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
            img_for_findContours = cv2.bilateralFilter(img_for_findContours, 11, 100, 150)
            # img_for_findContours = cv2.morphologyEx(img_for_findContours, cv2.MORPH_DILATE, np.ones(3, 3))
            contours, tree = cv2.findContours(img_for_findContours, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
            img_for_findContours_show = cv2.cvtColor(img_for_findContours, cv2.COLOR_GRAY2BGR)
            cv2.imwrite('Resualt.jpg', cv2.drawContours(img_for_findContours_show, contours, -1, (0, 0, 255)))

            for c in tqdm(sort_contours(contours, 0)):
                x, y, w, h = cv2.boundingRect(c)
                if w > 15 and h > 15:
                    image = img_for_findContours[y: y + h, x: x + w]
                    _, image = cv2.threshold(image, 50, 255, cv2.THRESH_BINARY)
                    text = image_to_string(image, recognition, 0.8)
                    image_list.append({
                        'x': x, 'y': y,
                        'image': image,
                        **text,
                    })
            # Sort
            resualt = []
            row = []
            for i, text in enumerate(image_list):
                if i == 0:
                    row.append(text)
                    continue
                last_text = image_list[i - 1]
                if last_text['y'] - 10 < text['y'] < last_text['y'] + 10:
                    row.append(text)
                else:
                    resualt.append(row)
                    row = [text]
            resualt.append(row)
            # Print
            for row in resualt:
                print(' '.join([f"{c['text']}" for c in sorted(row, key=lambda r: r['x'])]))

            cv2.imshow('show', img_for_findContours_show)
            cv2.waitKey(0)
    elif mode == '3':
        for image in images[2:3]:
            output = []
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            _, img_for_findContours = cv2.threshold(image, 200, 255, cv2.THRESH_BINARY_INV)
            # Rotate angle
            angle = find_rotate_angle(img_for_findContours)
            img_for_HoughLinesP = cv2.GaussianBlur(image, (3, 3), 10)
            _, img_for_HoughLinesP = cv2.threshold(img_for_HoughLinesP, 200, 255, cv2.THRESH_BINARY_INV)
            ver_lines = cv2.HoughLinesP(img_for_HoughLinesP, 1, np.pi, 100, minLineLength=50)
            hor_lines = cv2.HoughLinesP(img_for_HoughLinesP, 1, np.pi / 180, 200, minLineLength=100)
            for [[x0, y0, x1, y1]] in [*ver_lines, *hor_lines]:
                img_for_findContours = cv2.line(img_for_findContours, (x0, y0), (x1, y1), (0, 0, 0), 3)

            img_for_findContours = imutils.rotate(img_for_findContours, angle)

            img_for_findContours = cv2.bilateralFilter(img_for_findContours, 11, 100, 250)
            _, img_for_findContours = cv2.threshold(img_for_findContours, 150, 255, cv2.THRESH_BINARY)
            img_for_findContours = cv2.bilateralFilter(img_for_findContours, 9, 100, 100)
            # img_for_findContours = cv2.morphologyEx(img_for_findContours, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))
            img_for_findContours = cv2.cvtColor(img_for_findContours, cv2.COLOR_GRAY2BGR)
            data = pytesseract.image_to_data(img_for_findContours, lang='rus+eng', config=r'--psm 6 --oem 3',
                                             output_type=pytesseract.Output.DICT)
            for x, y, w, h, t, c in zip(data['left'], data['top'], data['width'], data['height'], data['text'], data['conf']):
                if float(c) < 0 : continue
                if float(c) < 80 :
                    contour_image = img_for_findContours[y: y + h, x: x + w]
                    contour_image = cv2.cvtColor(contour_image, cv2.COLOR_BGR2GRAY)
                    t = image_to_string(contour_image, recognition)[0]
                output.append({
                    'x': x, 'y': y, 'w': w, 'h': h, 'ocr_eps': c, 'ocr_text': t
                })
                # cv2.rectangle(img_for_findContours, (x, y), (x + w, y + h), (0, 0, 255))
                # cv2.putText(img_for_findContours, f'{t}', (x, y),cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255, 0))

            # Sort
            resualt = []
            row = []
            for i, text in enumerate(output):
                if i == 0:
                    row.append(text)
                    continue
                last_text = output[i - 1]
                if last_text['y'] - 20 < text['y'] < last_text['y'] + 20:
                    row.append(text)
                else:
                    resualt.append(row)
                    row = [text]
            resualt.append(row)
            # Print
            for row in resualt:
                print(' '.join([f"{c['ocr_text']}" for c in sorted(row, key=lambda r: r['x'])]))
            cv2.imwrite('Resualt.jpg', img_for_findContours)


@app.command()
def train():
    recognition = LetterRecognition(model_name='model_numbers_2dense_128_64')
    x, y = load_dataset('numbers')
    x = np.expand_dims(x, 3) / 255
    x_train, x_test, y_train, y_test = train_test_split(x, y)
    history = recognition.train(x_train, y_train, batch_size=32, epochs=100, summery=True)
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('Train')
    plt.show()


@app.command()
def test():
    recognition = LetterRecognition.load('model_numbers_2dense_128_64')
    x, y = load_dataset('numbers')
    x = np.expand_dims(x, 3) / 255
    x_train, x_test, y_train, y_test = train_test_split(x, y)
    predicts = recognition.predicts(x_test)
    mask = predicts == y_test
    x_false = x_test[~mask]
    print(f'{x_false.shape[0]}/{x_test.shape[0]}')
    # Evalute
    recognition.evaluate(x_test, y_test, 32)


if __name__ == '__main__':
    app()
