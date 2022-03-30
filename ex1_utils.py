"""
        '########:'##::::'##::::'##:::
         ##.....::. ##::'##:::'####:::
         ##::::::::. ##'##::::.. ##:::
         ######:::::. ###::::::: ##:::
         ##...:::::: ## ##:::::: ##:::
         ##:::::::: ##:. ##::::: ##:::
         ########: ##:::. ##::'######:
        ........::..:::::..:::......::
"""
from typing import List

import matplotlib.pyplot as plt
import numpy as np
import cv2

LOAD_GRAY_SCALE = 1
LOAD_RGB = 2


def myID() -> np.int:
    """
    Return my ID (not the friend's ID I copied from)
    :return: int
    """
    return 207765652


def imReadAndConvert(filename: str, representation: int) -> np.ndarray:
    """
    Reads an image, and returns the image converted as requested
    :param filename: The path to the image
    :param representation: GRAY_SCALE or RGB
    :return: The image object
    """
    img = cv2.imread(filename, )

    if representation == 1:
        img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        norm_img = img_gray / 255
        return norm_img
    else:
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        norm_img = img_rgb / 255
        return norm_img


def imDisplay(filename: str, representation: int):
    """
    Reads an image as RGB or GRAY_SCALE and displays it
    :param filename: The path to the image
    :param representation: GRAY_SCALE or RGB
    :return: None
    """
    img = imReadAndConvert(filename, representation)
    if representation == 1:
        plt.gray()
    plt.imshow(img)
    plt.show()


def transformRGB2YIQ(imgRGB: np.ndarray) -> np.ndarray:
    """
    Converts an RGB image to YIQ color space
    :param imgRGB: An Image in RGB
    :return: A YIQ in image color space
    """
    rgb_to_yiq = np.array([[0.299, 0.587, 0.114],
                           [0.596, -0.275, -0.321],
                           [0.212, -0.523, 0.311]])
    yiq_img = np.dot(imgRGB, rgb_to_yiq.transpose())
    return yiq_img


def transformYIQ2RGB(imgYIQ: np.ndarray) -> np.ndarray:
    """
    Converts an YIQ image to RGB color space
    :param imgYIQ: An Image in YIQ
    :return: A RGB in image color space
    """
    yiq_to_rgb = np.array([[0.299, 0.587, 0.114],
                           [0.596, -0.275, -0.321],
                           [0.212, -0.523, 0.311]])
    rgb_img = np.dot(imgYIQ, np.linalg.inv(yiq_to_rgb).transpose())
    return rgb_img


def hsitogramEqualize(imgOrig: np.ndarray) -> (np.ndarray, np.ndarray, np.ndarray):
    """
        Equalizes the histogram of an image
        :param imgOrig: Original Histogram
        :ret
    """
    is_rgb = False
    if len(imgOrig.shape) == 3:
        is_rgb = True
        yiq_img = transformRGB2YIQ(imgOrig)
        imgOrig = yiq_img[:, :, 0]
    # normalize back to [0,255]
    norm_image = cv2.normalize(imgOrig, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
    norm_image = norm_image.astype(np.uint8)
    histogram = np.histogram(norm_image.ravel(), bins=256)[0]
    cum_sum = np.cumsum(histogram)
    lut = (cum_sum / max(cum_sum)) * 255
    new_img = np.zeros_like(imgOrig)
    for i in range(256):
        new_img[norm_image == i] = lut[i]
    new_histogram = np.histogram(new_img.ravel(), bins=256)[0]
    if is_rgb == True:
        yiq_img[:, :, 0] = new_img/255.0
        new_img = transformYIQ2RGB(yiq_img)
    else:
        new_img /= 255.0
    return new_img, histogram, new_histogram


def quantizeImage(imOrig: np.ndarray, nQuant: int, nIter: int) -> (List[np.ndarray], List[float]):
    """
        Quantized an image in to **nQuant** colors
        :param imOrig: The original image (RGB or Gray scale)
        :param nQuant: Number of colors to quantize the image to
        :param nIter: Number of optimization loops
        :return: (List[qImage_i],List[error_i])
    """

