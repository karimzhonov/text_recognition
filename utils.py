import cv2.cv2 as cv2
import numpy as np
from pdf2image import convert_from_path


def pdf2numpy(path):
    """
    Convert pdf to numpy array
    :param path:
    :return: list of numpy array
    """
    images = convert_from_path(path)
    return [cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR) for image in images]


def recurent_findContours(img_for_findContours: np.array, param1: int, min_w: int, min_h: int, x0: int = 0, y0: int = 0,
                          output_shape: tuple[int, int] = None):
    """
    Recurent find contour
    :param img_for_findContours: Image
    :param param1: if contour weight small then param1 then appending to contour list else again finding contours
    :param min_w: min weight
    :param min_h: min height
    :param x0: cordinate of img_for_findContours
    :param y0: cordinate of img_for_findContours
    :param output_shape: shape of output contour (weight, height)
    :return: contour list of dict
    """
    img_for_findContours = 255 - img_for_findContours
    contours, tree, = cv2.findContours(img_for_findContours, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    return_contours = []
    if tree is not None or contours is not None:
        for contour, _ in zip(contours, tree[0]):
            x, y, w, h = cv2.boundingRect(contour)
            contour_img = img_for_findContours[y: y + h, x: x + w]
            if w < min_w or h < min_h:
                continue
            elif param1 > w:
                if output_shape:
                    contour_img = cv2.resize(contour_img, output_shape)
                return_contours.append({
                    'image': contour_img,
                    'x': x, 'y': y,
                })
            else:
                return_contours = [*return_contours,
                                   *recurent_findContours(contour_img, param1, min_w, min_h, x0 + x, y0 + y,
                                                          output_shape=output_shape)]
    else:
        if output_shape:
            img_for_findContours = cv2.resize(img_for_findContours, output_shape)
        return_contours.append({
            'image': img_for_findContours,
            'x': x0, 'y': y0,
        })
    return return_contours
