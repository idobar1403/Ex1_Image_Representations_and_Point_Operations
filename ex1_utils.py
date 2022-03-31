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
    img = cv2.imread(filename)
    # check whether the wanted representation is in
    # RGB or GRAYSCALE
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
    # using the function above to successfully read the img
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
    # matrix for the transformation
    rgb_to_yiq = np.array([[0.299, 0.587, 0.114],
                           [0.596, -0.275, -0.321],
                           [0.212, -0.523, 0.311]])
    # matrix multiplication
    yiq_img = np.dot(imgRGB, rgb_to_yiq.transpose())
    return yiq_img


def transformYIQ2RGB(imgYIQ: np.ndarray) -> np.ndarray:
    """
    Converts an YIQ image to RGB color space
    :param imgYIQ: An Image in YIQ
    :return: A RGB in image color space
    """
    # matrix for the transformation
    yiq_to_rgb = np.array([[0.299, 0.587, 0.114],
                           [0.596, -0.275, -0.321],
                           [0.212, -0.523, 0.311]])
    # matrix multiplication but with the inverse matrix
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
    # calculate histogram for the old img
    histogram = np.histogram(norm_image.ravel(), bins=256)[0]
    cum_sum = np.cumsum(histogram)
    lut = (cum_sum / max(cum_sum)) * 255
    new_img = np.zeros_like(imgOrig)
    # calculate the lut
    for i in range(256):
        new_img[norm_image == i] = lut[i]
    # flatten the new img in order to make histogram
    new_histogram = np.histogram(new_img.ravel(), bins=256)[0]
    if is_rgb == True:
        # if the img was RGB then transfer it back
        yiq_img[:, :, 0] = new_img / 255.0
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
    is_rgb = False
    # check if the img is in RGB colorspace
    if len(imOrig.shape) == 3:
        is_rgb = True
        yiq_img = transformRGB2YIQ(imOrig)  # transform to YIQ
        imOrig = yiq_img[:, :, 0]  # take only the Y from the YIQ
    images, errors = quantize_one_dim(imOrig, nQuant, nIter)
    if is_rgb:
        new_images = []
        for img in images:
            # set back the new Y
            yiq_img[:, :, 0] = img
            curr_img = transformYIQ2RGB(yiq_img)
            new_images.append(curr_img)
        return new_images, errors
    else:
        return images, errors


def quantize_one_dim(imgOrig: np.ndarray, nQuant: int, nIter: int) -> (List[np.ndarray], List[float]):
    mses = []  # the error for each iteration
    new_images = []
    img_hist = np.histogram((imgOrig.ravel() * 255), bins=256)[0]
    boarder_z = []  # the boarders
    # select the boundaries for the start of the iterations
    cum_sum = np.cumsum(img_hist)
    boarder_z = [0]  # adding 0 as first boarder
    num_of_pixels = img_hist.sum() / nQuant  # calculate number of pixels that should be between two boarders
    curr_sum = 0
    # select the best boundaries for the start of the iterations
    for i in range(0, 256):
        # check if passed the amount of pixels for the current boundary
        curr_sum += img_hist[i]
        if curr_sum >= num_of_pixels:
            boarder_z.append(i + 1)
            curr_sum = 0
    # add 256 as the last boundary
    if 255 not in boarder_z:
        boarder_z.append(255)
    for i in range(nIter):
        averages = []
        imgs = np.zeros_like(imgOrig)
        for j in range(nQuant):
            # get the number of pixels in the histogram between j and j+1
            pixels_in_range = img_hist[boarder_z[j]:boarder_z[j + 1]]
            index = np.arange(int(boarder_z[j]), int(boarder_z[j + 1]))
            # get the average of all the pixels that between the boundaries
            avg = (index * pixels_in_range).sum() / (pixels_in_range.sum()).astype(int)
            averages.append(avg)
        # set every pixel inside boundaries as the average
        for a_v_g in range(len(averages)):
            imgs[imgOrig > boarder_z[a_v_g] / 255] = averages[a_v_g]
        # set new and better boundaries according to the averages in every boundary
        for boarder in range(1, len(boarder_z) - 1):
            boarder_z[boarder] = int((averages[boarder - 1] + averages[boarder]) / 2)
        # calculate the mse between the old and the new img
        curr_ms_error = np.sqrt((imgOrig * 255 - imgs) ** 2).mean()
        mses.append(curr_ms_error)
        new_images.append(imgs / 255)

    return new_images, mses
